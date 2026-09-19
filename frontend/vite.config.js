import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ingest': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/uploads': 'http://localhost:8000',
    },
  },
})
