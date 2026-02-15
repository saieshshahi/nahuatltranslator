import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const repoName = "nahuatltranslator";
const base =
  process.env.VITE_BASE_PATH ||
  (process.env.GITHUB_PAGES === "true" ? `/${repoName}/` : "/");

export default defineConfig({
  plugins: [react()],
  base,
  server: {
    port: 5173,
    strictPort: true,
  },
});
