import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the browser talks to Vite (port 5173), and Vite forwards
// every /api call to the Python backend (port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
