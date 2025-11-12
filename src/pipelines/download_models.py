import logging

from sentence_transformers import SentenceTransformer

from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Download and cache the embedding model."""
    logger.info(f"Downloading embedding model: {Config.EMBEDDING_MODEL_NAME}")

    try:
        model_cache_path = Config.get_model_cache_path(Config.EMBEDDING_MODEL_NAME)

        SentenceTransformer(
            Config.EMBEDDING_MODEL_NAME,
            cache_folder=str(Config.MODELS_PATH),
            token=Config.HUGGINGFACE_TOKEN,
        )

        logger.info(f"Embedding model downloaded successfully to {model_cache_path}")

    except Exception as error:
        logger.error(f"Failed to download embedding model: {error}")
        raise


if __name__ == "__main__":
    main()
