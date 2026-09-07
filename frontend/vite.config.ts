import path from 'node:path';
import { defineConfig, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

// Backend origin the `/api/*` proxy forwards to. In production (`vite preview`
// on Railway) this is read at process start from BACKEND_URL so cookies stay
// first-party; locally it falls back to the dev backend.
const backendUrl = (process.env.BACKEND_URL ?? 'http://localhost:8000').replace(/\/+$/, '');

const apiProxy: Record<string, ProxyOptions> = {
  '/api': {
    target: backendUrl,
    changeOrigin: true,
    secure: true,
    rewrite: (p) => p.replace(/^\/api/, ''),
  },
};

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 3000,
    proxy: apiProxy,
  },
  preview: {
    allowedHosts: true,
    proxy: apiProxy,
  },
});
