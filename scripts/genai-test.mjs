import { GoogleGenAI } from "@google/genai";

const apiKey =
  process.env.GEMINI_API_KEY ||
  process.env.GOOGLE_API_KEY ||
  process.env.VITE_GEMINI_API_KEY;

if (!apiKey) {
  console.error(
    "API key topilmadi. `GEMINI_API_KEY` (yoki `GOOGLE_API_KEY`) ni terminal env ga qo'ying.",
  );
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

async function main() {
  const model = process.env.GEMINI_MODEL || "gemini-3-flash-preview";
  const response = await ai.models.generateContent({
    model,
    contents: "Explain how AI works in a few words",
  });
  console.log(response.text);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`GenAI xato: ${message}`);
  process.exit(1);
});
