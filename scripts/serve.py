"""Run the FastAPI app via uvicorn. Reads gold features from disk."""

import uvicorn

from kadastra.composition_root import create_app
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
