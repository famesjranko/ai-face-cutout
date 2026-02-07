import logging
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level="info")
