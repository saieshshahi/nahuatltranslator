import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages requires a base path like /REPO_NAME/.
// We set it at build time via VITE_BASE_PATH (GitHub Action fills this automatically).
const base = process.env.VITE_BASE_PATH || '/'

export default defineConfig({
  plugins: [react()],
  base,
})
