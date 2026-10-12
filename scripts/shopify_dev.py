"""Run the existing FastAPI application using Shopify CLI's assigned port."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import uvicorn

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Read CLI variables before the app loads .env (which contains production settings).
    port = int(os.getenv("PORT") or os.getenv("FRONTEND_PORT") or "8001")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(ROOT / "app")],
    )


if __name__ == "__main__":
    main()
