import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

const backend = process.env.ADMIN_BACKEND_URL || "http://127.0.0.1:8001";
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/admin/assets/",
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  build: { outDir: "../app/static/admin", emptyOutDir: true },
  server: {
    proxy: {
      "^/admin/(?!assets(?:/|$))": {
        target: backend,
        bypass(request) {
          if (request.method === "GET" && /^\/admin\/(login|tasks|styles|config)(?:\?|$)/.test(request.url || "")) {
            return "/admin/assets/index.html";
          }
        },
      },
      "/health": backend,
      "/taskfile": backend,
    },
  },
});
