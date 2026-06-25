import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Why bind to 127.0.0.1 explicitly:
//   On Node 17+ the string "localhost" resolves to ::1 first, which makes
//   Vite IPv6-only. Browsers on Windows often hit 127.0.0.1 first and the
//   page appears stuck / "not updating" because HMR over WebSocket times
//   out. Pinning host to 127.0.0.1 keeps the dev server reachable on
//   IPv4, which is what every link in our docs uses.
//
// Why the /api proxy:
//   We just removed the hardcoded `http://127.0.0.1:8765` API base from
//   src/hooks/api.ts. Production traffic stays on the same origin as
//   the SPA bundle that FastAPI serves; for dev we forward /api/** here
//   so the same code path works against the existing 8765 backend.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8765'

  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      port: 5173,
      strictPort: true,
      hmr: {
        host: '127.0.0.1',
        protocol: 'ws',
      },
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: false,
          ws: false,
        },
      },
    },
  }
})
