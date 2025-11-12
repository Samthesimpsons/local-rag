import logging
import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Config:
    """Configuration class for the RAG application."""

    PROJECT_ROOT: Path = Path(__file__).parent.parent
    MODELS_PATH: Path = PROJECT_ROOT / os.getenv("MODELS_PATH")  # type: ignore
    DOCUMENTS_PATH: Path = PROJECT_ROOT / os.getenv("DOCUMENTS_PATH")  # type: ignore
    CHROMA_DB_PATH: Path = PROJECT_ROOT / os.getenv("CHROMA_DB_PATH")  # type: ignore

    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN")  # type: ignore
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME")  # type: ignore

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")  # type: ignore
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")  # type: ignore
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")  # type: ignore

    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME")  # type: ignore
    ANTHROPIC_MODEL_NAME: str = os.getenv("ANTHROPIC_MODEL_NAME")  # type: ignore
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME")  # type: ignore

    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME")  # type: ignore

    @classmethod
    def validate_configuration(cls) -> None:
        """Validate that required configuration is present."""
        logger.info("Validating configuration...")

        cls.MODELS_PATH.mkdir(parents=True, exist_ok=True)
        cls.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

        logger.info("Configuration validated. LLM provider will be selected from UI.")

    @classmethod
    def get_model_cache_path(cls, model_name: str) -> Path:
        """Get the local cache path for a specific model."""
        safe_name = model_name.replace("/", "_")
        return cls.MODELS_PATH / safe_name


Config.validate_configuration()
