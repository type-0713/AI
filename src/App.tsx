import { useEffect, useRef, useState } from "react";
import { collection, doc, onSnapshot, setDoc } from "firebase/firestore";
import { db } from "./firebase";

type Sender = "user" | "bot";

interface Message {
  id: string;
  text: string;
  sender: Sender;
}

interface ChatItem {
  id: string;
  title: string;
  lastMessage: string;
  updatedAt: number;
}

interface ChatDocData {
  title?: string;
  lastMessage?: string;
  updatedAt?: number;
  createdAt?: number;
  messages?: Message[];
}

interface GeminiResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{ text?: string }>;
    };
  }>;
}

interface GeminiErrorResponse {
  error?: {
    message?: string;
  };
}

interface GeminiModelItem {
  name?: string;
  supportedGenerationMethods?: string[];
}

interface GeminiModelsResponse {
  models?: GeminiModelItem[];
}

interface LocalCache {
  chatList: ChatItem[];
  messagesMap: Record<string, Message[]>;
  activeChatId: string | null;
}

const GEMINI_KEY_STORAGE_KEY = "gemini_api_key";

const MODEL_IDS = [
  "gemini-2.5-flash",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-flash-latest",
];
const API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models";

const RETRY_DELAY_MS = 1200;
const REQUEST_TIMEOUT_MS = 30000;
const LOCAL_CACHE_KEY = "chatbot_cache_v2";
const ASSISTANT_INSTRUCTION =
  "Siz yordamchi AI'siz. Har doim adabiy o'zbek tilida, ravon va xatosiz yozing. Juda qisqa bo'lmang, lekin ortiqcha cho'zmang. Savolga aniq javob bering, keraksiz takrorlar va aralash til ishlatmang. Ilmiy savollarda aniq tushuntiring, oddiy savollarda sodda va tushunarli javob bering.  Har doim insonlarga yordam berishga harakat qiling va muloyim bo'ling. Hamma sizdan hursand bo'lsin!";

const toPreview = (text: string) => {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return "Bo'sh xabar";
  }
  return cleaned.length > 50 ? `${cleaned.slice(0, 50)}...` : cleaned;
};

const chatTitleFromMessages = (messages: Message[]) => {
  const firstUserMessage = messages.find((msg) => msg.sender === "user");
  if (!firstUserMessage) {
    return "New chat";
  }
  return toPreview(firstUserMessage.text);
};

const areMessagesEqual = (left: Message[], right: Message[]) => {
  if (left.length !== right.length) {
    return false;
  }
  for (let i = 0; i < left.length; i += 1) {
    if (
      left[i].id !== right[i].id ||
      left[i].text !== right[i].text ||
      left[i].sender !== right[i].sender
    ) {
      return false;
    }
  }
  return true;
};

const normalizeMessages = (input: unknown): Message[] => {
  if (!Array.isArray(input)) {
    return [];
  }

  return input.filter(
    (msg): msg is Message =>
      typeof msg === "object" &&
      msg !== null &&
      typeof (msg as Message).id === "string" &&
      typeof (msg as Message).text === "string" &&
      (((msg as Message).sender === "user") || (msg as Message).sender === "bot"),
  );
};

const cleanModelText = (input: string) =>
  input
    .replace(/\uFFFD/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();

const buildFallbackReply = (userText: string, reason: string) =>
  `Hozir API bilan ulanishda muammo bor, lekin savolingizni oldim:\n"${userText}"\n\nXato: ${reason}\n\nIltimos API key ruxsatlarini tekshiring yoki 10-20 soniyadan keyin qayta yuboring.`;

const fetchWithTimeout = async (url: string, init: RequestInit) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
};

const getStoredGeminiApiKey = () => {
  if (typeof window === "undefined") {
    return "";
  }
  return (window.localStorage.getItem(GEMINI_KEY_STORAGE_KEY) || "").trim();
};

const getGeminiApiKey = () =>
  ((import.meta.env.VITE_GEMINI_API_KEY as string | undefined) || getStoredGeminiApiKey()).trim();

const requestGeminiApiKey = () => {
  if (typeof window === "undefined") {
    return "";
  }

  const entered = window.prompt(
    "Gemini API key kiriting (bir marta saqlanadi). .env ishlatmoqchi bo'lsangiz VITE_GEMINI_API_KEY ni sozlang.",
  );
  const key = (entered || "").trim();
  if (key) {
    window.localStorage.setItem(GEMINI_KEY_STORAGE_KEY, key);
  }
  return key;
};

const buildModelUrl = (modelId: string, includeQueryKey: boolean, apiKey: string) => {
  const url = new URL(`${API_BASE_URL}/${modelId}:generateContent`);
  if (includeQueryKey) {
    url.searchParams.set("key", apiKey);
  }
  return url.toString();
};

const buildListModelsUrl = (includeQueryKey: boolean, apiKey: string) => {
  const url = new URL(API_BASE_URL);
  if (includeQueryKey) {
    url.searchParams.set("key", apiKey);
  }
  return url.toString();
};

const cachedModelIdsByKey: Record<string, string[]> = {};

const getRuntimeModelIds = async (apiKey: string) => {
  if (!apiKey) {
    return MODEL_IDS;
  }

  if (cachedModelIdsByKey[apiKey]) {
    return cachedModelIdsByKey[apiKey];
  }

  const variants = [
    {
      url: buildListModelsUrl(true, apiKey),
      headers: { "Content-Type": "application/json" } as Record<string, string>,
    },
    {
      url: buildListModelsUrl(false, apiKey),
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      } as Record<string, string>,
    },
  ];

  for (const variant of variants) {
    try {
      const response = await fetchWithTimeout(variant.url, {
        method: "GET",
        headers: variant.headers,
      });

      if (!response.ok) {
        continue;
      }

      const data = (await response.json()) as GeminiModelsResponse;
      const supported = (data.models || [])
        .filter((model) => (model.supportedGenerationMethods || []).includes("generateContent"))
        .map((model) => model.name?.replace("models/", ""))
        .filter((name): name is string => Boolean(name));

      if (supported.length > 0) {
        const preferred = MODEL_IDS.filter((id) => supported.includes(id));
        cachedModelIdsByKey[apiKey] = preferred.length > 0 ? preferred : supported;
        return cachedModelIdsByKey[apiKey];
      }
    } catch {
      // Continue with next variant.
    }
  }

  cachedModelIdsByKey[apiKey] = MODEL_IDS;
  return cachedModelIdsByKey[apiKey];
};

export default function App() {
  const [chatList, setChatList] = useState<ChatItem[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [messagesMap, setMessagesMap] = useState<Record<string, Message[]>>({});
  const [input, setInput] = useState("");
  const [pendingChats, setPendingChats] = useState<Record<string, boolean>>({});
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesMapRef = useRef<Record<string, Message[]>>({});
  const activeChatPending = activeChatId ? Boolean(pendingChats[activeChatId]) : false;
  const closeSidebarOnMobile = () => {
    if (window.matchMedia("(max-width: 900px)").matches) {
      setIsSidebarOpen(false);
    }
  };

  useEffect(() => {
    messagesMapRef.current = messagesMap;
  }, [messagesMap]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(LOCAL_CACHE_KEY);
      if (!raw) {
        return;
      }

      const parsed = JSON.parse(raw) as LocalCache;
      if (Array.isArray(parsed.chatList)) {
        setChatList(parsed.chatList);
      }

      if (parsed.messagesMap && typeof parsed.messagesMap === "object") {
        const nextMap: Record<string, Message[]> = {};
        Object.entries(parsed.messagesMap).forEach(([chatId, list]) => {
          nextMap[chatId] = normalizeMessages(list);
        });
        setMessagesMap(nextMap);
      }

      if (typeof parsed.activeChatId === "string" || parsed.activeChatId === null) {
        setActiveChatId(parsed.activeChatId);
        if (parsed.activeChatId && parsed.messagesMap?.[parsed.activeChatId]) {
          setMessages(normalizeMessages(parsed.messagesMap[parsed.activeChatId]));
        }
      }
    } catch {
      localStorage.removeItem(LOCAL_CACHE_KEY);
    }
  }, []);

  useEffect(() => {
    const cache: LocalCache = {
      chatList,
      messagesMap,
      activeChatId,
    };
    localStorage.setItem(LOCAL_CACHE_KEY, JSON.stringify(cache));
  }, [chatList, messagesMap, activeChatId]);

  useEffect(() => {
    const chatsRef = collection(db, "chats");

    const unsubscribe = onSnapshot(
      chatsRef,
      (snapshot) => {
        const nextList = snapshot.docs
          .map((docSnap) => {
            const data = docSnap.data() as ChatDocData;
            const updatedAt =
              typeof data.updatedAt === "number"
                ? data.updatedAt
                : typeof data.createdAt === "number"
                  ? data.createdAt
                  : 0;

            return {
              id: docSnap.id,
              title: data.title || "New chat",
              lastMessage: data.lastMessage || "",
              updatedAt,
            } as ChatItem;
          })
          .sort((a, b) => b.updatedAt - a.updatedAt);

        setChatList(nextList);
        setActiveChatId((prev) => {
          if (prev && nextList.some((item) => item.id === prev)) {
            return prev;
          }
          return nextList[0]?.id ?? prev ?? null;
        });
        setStatusText("");
      },
      () => {
        setStatusText("Firebase ulanmagan. Chat lokal keshdan ishlayapti.");
      },
    );

    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (!activeChatId) {
      setMessages([]);
      return;
    }

    const cachedMessages = messagesMapRef.current[activeChatId];
    if (cachedMessages) {
      setMessages(cachedMessages);
    }

    const chatRef = doc(db, "chats", activeChatId);
    const unsubscribe = onSnapshot(
      chatRef,
      (snapshot) => {
        if (!snapshot.exists()) {
          return;
        }

        const data = snapshot.data() as ChatDocData;
        const nextMessages = normalizeMessages(data.messages);
        setMessages(nextMessages);
        setMessagesMap((prev) => {
          const current = prev[activeChatId] ?? [];
          if (areMessagesEqual(current, nextMessages)) {
            return prev;
          }
          return { ...prev, [activeChatId]: nextMessages };
        });
      },
      () => {
        // Keep local cache messages if firestore subscription fails.
      },
    );

    return () => unsubscribe();
  }, [activeChatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeChatPending]);

  useEffect(() => {
    if (!activeChatId) {
      return;
    }
    setMessagesMap((prev) => {
      const current = prev[activeChatId] ?? [];
      if (areMessagesEqual(current, messages)) {
        return prev;
      }
      return { ...prev, [activeChatId]: messages };
    });
  }, [activeChatId, messages]);

  const updateLocalChatMeta = (chatId: string, nextMessages: Message[]) => {
    const updatedAt = Date.now();
    const title = chatTitleFromMessages(nextMessages);
    const lastMessage = nextMessages.length > 0 ? toPreview(nextMessages[nextMessages.length - 1].text) : "";

    setMessagesMap((prev) => ({ ...prev, [chatId]: nextMessages }));
    setChatList((prev) => {
      const without = prev.filter((item) => item.id !== chatId);
      return [{ id: chatId, title, lastMessage, updatedAt }, ...without].sort((a, b) => b.updatedAt - a.updatedAt);
    });
  };

  const persistChatBestEffort = async (chatId: string, nextMessages: Message[]) => {
    const updatedAt = Date.now();

    try {
      await setDoc(
        doc(db, "chats", chatId),
        {
          title: chatTitleFromMessages(nextMessages),
          lastMessage: nextMessages.length > 0 ? toPreview(nextMessages[nextMessages.length - 1].text) : "",
          updatedAt,
          messages: nextMessages,
        },
        { merge: true },
      );
    } catch {
      setStatusText("Firebasega saqlashda xato. Xabarlar lokal keshta saqlandi.");
    }
  };

  const createChat = async () => {
    const localId = crypto.randomUUID();
    const now = Date.now();

    setMessages([]);
    setActiveChatId(localId);
    setMessagesMap((prev) => ({ ...prev, [localId]: [] }));
    setChatList((prev) => [{ id: localId, title: "New chat", lastMessage: "", updatedAt: now }, ...prev]);

    try {
      const chatRef = doc(collection(db, "chats"));
      await setDoc(chatRef, {
        title: "New chat",
        lastMessage: "",
        createdAt: now,
        updatedAt: now,
        messages: [],
      });

      setActiveChatId(chatRef.id);
      setMessagesMap((prev) => {
        const { [localId]: localDraftEntry, ...rest } = prev;
        void localDraftEntry;
        return { ...rest, [chatRef.id]: [] };
      });
      setChatList((prev) => {
        const filtered = prev.filter((item) => item.id !== localId);
        return [{ id: chatRef.id, title: "New chat", lastMessage: "", updatedAt: now }, ...filtered];
      });
      setStatusText("");
      return chatRef.id;
    } catch {
      setStatusText("Firebasega yangi chat yozilmadi. Lokal chat yaratildi.");
      return localId;
    }
  };

  const getGeminiReply = async (userText: string) => {
    let apiKey = getGeminiApiKey();
    if (!apiKey) {
      apiKey = requestGeminiApiKey();
    }

    if (!apiKey) {
      const missingKeyReason =
        "Gemini API key topilmadi (`VITE_GEMINI_API_KEY` yoki saqlangan local key yo'q).";
      setStatusText(missingKeyReason);
      return buildFallbackReply(userText, missingKeyReason);
    }

    let finalError = "AI javob qaytarmadi.";
    const runtimeModelIds = await getRuntimeModelIds(apiKey);

    for (let modelIndex = 0; modelIndex < runtimeModelIds.length; modelIndex += 1) {
      const modelId = runtimeModelIds[modelIndex];
      const requestVariants = [
        {
          label: "query",
          url: buildModelUrl(modelId, true, apiKey),
          headers: { "Content-Type": "application/json" } as Record<string, string>,
        },
        {
          label: "header",
          url: buildModelUrl(modelId, false, apiKey),
          headers: {
            "Content-Type": "application/json",
            "x-goog-api-key": apiKey,
          } as Record<string, string>,
        },
      ];

      for (const variant of requestVariants) {
        for (let attempt = 1; attempt <= 2; attempt += 1) {
          let response: Response;
          try {
            response = await fetchWithTimeout(variant.url, {
              method: "POST",
              headers: variant.headers,
              body: JSON.stringify({
                system_instruction: {
                  parts: [{ text: ASSISTANT_INSTRUCTION }],
                },
                generationConfig: {
                  temperature: 0.35,
                  topP: 0.9,
                  maxOutputTokens: 1024,
                },
                contents: [
                  {
                    parts: [{ text: userText }],
                  },
                ],
              }),
            });
          } catch (error) {
            const message =
              error instanceof Error && error.message
                ? error.message
                : "Network xato";
            finalError = `[${modelId}/${variant.label}] ${message}`;
            break;
          }

          if (response.ok) {
            const data = (await response.json()) as GeminiResponse;
            const text = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
            return text ? cleanModelText(text) : "Javob olinmadi.";
          }

          let errMessage = finalError;
          try {
            const err = (await response.json()) as GeminiErrorResponse;
            errMessage = err.error?.message || finalError;
          } catch {
            errMessage = `${response.status} ${response.statusText}`.trim();
          }

          const leakedKeyDetected = errMessage.toLowerCase().includes("reported as leaked");
          finalError = leakedKeyDetected
            ? `[${modelId}/${variant.label}] API key oshkor bo'lgani uchun bloklangan. Yangi key yarating.`
            : `[${modelId}/${variant.label}] ${errMessage}`;
          const isHighDemand =
            response.status === 429 ||
            response.status === 503 ||
            errMessage.toLowerCase().includes("high demand");

          if (!isHighDemand || attempt === 2) {
            break;
          }

          await new Promise((resolve) => {
            setTimeout(resolve, RETRY_DELAY_MS * attempt);
          });
        }
      }
    }

    setStatusText(finalError);
    return buildFallbackReply(userText, finalError);
  };

  const sendMessage = async () => {
    if (!input.trim()) {
      return;
    }

    const userText = input.trim();
    let requestChatId: string | null = null;

    try {
      const chatId = activeChatId || (await createChat());
      requestChatId = chatId;
      if (pendingChats[chatId]) {
        setStatusText("AI oldingi xabarga javob yozmoqda, biroz kuting.");
        return;
      }

      setStatusText("");
      setInput("");
      setPendingChats((prev) => ({ ...prev, [chatId]: true }));
      const baseMessages = activeChatId === chatId ? messages : messagesMap[chatId] ?? [];

      const userMessage: Message = {
        id: crypto.randomUUID(),
        text: userText,
        sender: "user",
      };

      const withUser = [...baseMessages, userMessage];
      setActiveChatId(chatId);
      setMessages(withUser);
      updateLocalChatMeta(chatId, withUser);
      void persistChatBestEffort(chatId, withUser);

      const botText = await getGeminiReply(userText);
      const botMessage: Message = {
        id: crypto.randomUUID(),
        text: botText,
        sender: "bot",
      };

      const withBot = [...withUser, botMessage];
      setMessages(withBot);
      updateLocalChatMeta(chatId, withBot);
      void persistChatBestEffort(chatId, withBot);
    } catch (error) {
      const errorText =
        error instanceof Error && error.message
          ? error.message
          : "AI bilan bog'lanishda xato. Internet yoki API sozlamasini tekshiring.";

      const botMessage: Message = {
        id: crypto.randomUUID(),
        text: errorText,
        sender: "bot",
      };

      const chatId = requestChatId;
      if (chatId) {
        const withError = [...(messagesMap[chatId] ?? messages), botMessage];
        setMessages(withError);
        updateLocalChatMeta(chatId, withError);
        void persistChatBestEffort(chatId, withError);
      } else {
        setMessages((prev) => [...prev, botMessage]);
      }
    } finally {
      if (requestChatId) {
        setPendingChats((prev) => ({ ...prev, [requestChatId as string]: false }));
      }
    }
  };

  const copyText = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 1500);
    } catch {
      setStatusText("Nusxa olish uchun browser ruxsatini yoqing.");
    }
  };

  return (
    <div className={`app-shell ${isSidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <aside className={`chat-sidebar ${isSidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-top">
          <img src="/favicon.png" alt="AI" className="sidebar-logo" />
          <div className="sidebar-top-actions">
            <button
              type="button"
              className="icon-ghost-btn mobile-only"
              onClick={() => setIsSidebarOpen(false)}
              aria-label="Sidebar yopish"
              title="Yopish"
            >
              ×
            </button>
          </div>
        </div>

        <div className="sidebar-actions">
          <button
            type="button"
            className="sidebar-action-btn"
            onClick={() => {
              void createChat();
              closeSidebarOnMobile();
            }}
          >
            New chat
          </button>
        </div>

        <p className="sidebar-section-title">Your chats</p>
        <div className="chat-list">
          {chatList.length === 0 ? (
            <p className="sidebar-empty">Chatlar hozircha yo'q</p>
          ) : (
            chatList.map((chat) => (
              <button
                key={chat.id}
                type="button"
                className={`chat-list-item ${chat.id === activeChatId ? "active" : ""}`}
                onClick={() => {
                  setActiveChatId(chat.id);
                  closeSidebarOnMobile();
                }}
              >
                <span className="chat-title">{chat.title}</span>
                <span className="chat-preview">{chat.lastMessage || "Yangi chat"}</span>
              </button>
            ))
          )}
        </div>
      </aside>

      <section className="chat-layout">
        <header className="chat-header">
          <button
            type="button"
            className="icon-ghost-btn mobile-only"
            onClick={() => setIsSidebarOpen(true)}
            aria-label="Sidebar ochish"
            title="Menyu"
          >
            =
          </button>
          <h2>AI Chat</h2>
        </header>

        <main className="chat-body">
          {statusText && <div className="typing">{statusText}</div>}

          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Chat boshlang</h2>
              <p>Istalgan tilda yozishingiz mumkin.</p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message-row ${msg.sender === "user" ? "message-right" : "message-left"}`}
            >
              <article className={`message ${msg.sender}`}>
                <p>{msg.text}</p>
                {msg.sender === "bot" && (
                  <button
                    type="button"
                    className={`copy-icon-btn ${copiedMessageId === msg.id ? "copied" : ""}`}
                    aria-label="Nusxa olish"
                    title={copiedMessageId === msg.id ? "Nusxalandi" : "Nusxa olish"}
                    onClick={() => void copyText(msg.text, msg.id)}
                  >
                    <span className="copy-icon" aria-hidden="true">
                      <span className="copy-icon-back" />
                      <span className="copy-icon-front" />
                    </span>
                  </button>
                )}
              </article>
            </div>
          ))}

          {activeChatPending && <div className="typing">AI yozmoqda...</div>}
          <div ref={messagesEndRef} />
        </main>

        <footer className="chat-input-wrap">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            placeholder="Xabar yozing..."
            rows={2}
          />
          <button
            type="button"
            className={`send-icon-btn ${activeChatPending ? "pending" : ""}`}
            onClick={() => void sendMessage()}
            disabled={activeChatPending}
            aria-label={activeChatPending ? "Yozishni kuting" : "Yuborish"}
            title={activeChatPending ? "AI yozmoqda" : "Yuborish"}
          >
            {activeChatPending ? (
              <span className="send-stop-icon" aria-hidden="true" />
            ) : (
              <span className="send-wave-icon" aria-hidden="true">
                <span />
                <span />
                <span />
                <span />
                <span />
              </span>
            )}
          </button>
        </footer>
      </section>
    </div>
  );
}
