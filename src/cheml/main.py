import logging

import uvicorn
from fastapi import FastAPI


from cheml.app_logging import setup_logging
from cheml.routes.manufacturing_quality import router as manufacturing_router
import mlflow

app = FastAPI()
app.include_router(manufacturing_router, prefix="/manufacturing-quality")


@app.get("/")
async def root():
    return {"message": "Simulation Model API is running!"}


def initialize_app(app: FastAPI) -> None:
    # Perform any necessary initialization here
    setup_logging()
    logger = logging.getLogger("Initialization")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info("MLflow tracking URI set to http://127.0.0.1:5000")

    logger.info("App initialized successfully.")


initialize_app(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
