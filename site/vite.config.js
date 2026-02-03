import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5002,
    allowedHosts: ["one-llm-4-all.purecore.codes"],
  },
  preview: {
    host: true,
    port: 5002,
    allowedHosts: ["one-llm-4-all.purecore.codes"],
  },
});
