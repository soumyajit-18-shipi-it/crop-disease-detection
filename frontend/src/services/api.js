import axios from "axios";

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000
});

function normalizeError(error) {
  if (error.response?.data?.detail) {
    return new Error(Array.isArray(error.response.data.detail) ? "Request validation failed." : error.response.data.detail);
  }
  if (error.code === "ECONNABORTED") {
    return new Error("The server took too long to respond.");
  }
  if (!error.response) {
    return new Error("Cannot reach the backend. Check that FastAPI is running.");
  }
  return new Error("Something went wrong while processing the request.");
}

export async function predictDisease(file) {
  try {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/predict", formData);
    return response.data;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDiseaseInfo(className) {
  try {
    const response = await api.get(`/disease/${encodeURIComponent(className)}`);
    return response.data;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getHistory(limit = 50) {
  try {
    const response = await api.get("/history", { params: { limit } });
    return response.data;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getHealth() {
  try {
    const response = await api.get("/health");
    return response.data;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function sendFeedback(payload) {
  try {
    const response = await api.post("/feedback", payload);
    return response.data;
  } catch (error) {
    throw normalizeError(error);
  }
}
