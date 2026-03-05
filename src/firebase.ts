import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyD_tDypWTr27pSPGj1bmYXuQ_OD7iLr1iw",
  authDomain: "chatbot-8715f.firebaseapp.com",
  projectId: "chatbot-8715f",
  storageBucket: "chatbot-8715f.firebasestorage.app",
  messagingSenderId: "928644659265",
  appId: "1:928644659265:web:6e0bb9f6af603124e3eec0",
  measurementId: "G-MYLWM2ZTPB",
};

const app = initializeApp(firebaseConfig);

export const db = getFirestore(app);
