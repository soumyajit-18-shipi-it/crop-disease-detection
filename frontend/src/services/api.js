import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
});

export async function predictDisease(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return response.data;
}

export async function getDiseaseInfo(className) {
  const response = await api.get(`/disease/${encodeURIComponent(className)}`);
  return response.data;
}
