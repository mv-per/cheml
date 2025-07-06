import logging
from logging.handlers import RotatingFileHandler
import os
from cheml.configs import BASE_PATH


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    """
    base_path = os.path.join(BASE_PATH, "logs")
    os.makedirs(base_path, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(base_path, "app.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[file_handler, stream_handler],
    )

    # Overwrite FastAPI/uvicorn loggers to use the same handlers and level
    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ):
        logger = logging.getLogger(logger_name)
        logger.handlers = [file_handler, stream_handler]
        logger.setLevel(level)
        logger.propagate = False
