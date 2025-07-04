import logging

import uvicorn
from fastapi import FastAPI


from cheml.app_logging import setup_logging


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Simulation Model API is running!"}


def initialize_app(app: FastAPI) -> None:
    # Perform any necessary initialization here
    setup_logging()
    logger = logging.getLogger("Initialization")

    logger.info("App initialized successfully.")


initialize_app(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
