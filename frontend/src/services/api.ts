import axios from "axios";

// Use `VITE_API_URL` when provided (set in Vercel). In production fallback to the Render URL.
const baseURL = import.meta.env.VITE_API_URL || "https://ai-detection-system-3.onrender.com";

const API = axios.create({
  baseURL,
});

export default API;