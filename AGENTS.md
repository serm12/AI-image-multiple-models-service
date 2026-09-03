# Codex Project Notes

- This is the backend for the AI image API.
- Default production deploy target: `image-api.shopupup.com`.
- Production is served through BaoTa/Nginx reverse proxy to `http://127.0.0.1:8002`.
- BaoTa MCP helper project: `C:\Users\yt640\Documents\mcp-server\mcp-bt-panel`.
- When the user asks to push/deploy this backend, push this repo first, then verify the BaoTa deployment for `image-api.shopupup.com`.
- Deployment verification should include:
  - `https://image-api.shopupup.com/health`
  - Docker container `ai-image-api` is running and healthy
  - Nginx config test passes
- Do not treat `aiapiroute.com` as the default backend deployment domain unless the user explicitly asks for it.
