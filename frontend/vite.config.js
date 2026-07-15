import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const target = process.env.VITE_DEV_API_TARGET;
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: target
        ? {
            "/auth": target,
            "/dashboard": target,
            "/history": target,
            "/predict": target,
            "/feedback": target,
            "/health": target,
            "/classes": target,
            "/disease": target,
          }
        : undefined,
    },
  };
});
