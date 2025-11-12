import logging
from abc import ABC, abstractmethod

import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI

from src.config import Config, LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """Base class for LLM service implementations."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAILLMService(BaseLLMService):
    """LLM service using OpenAI API."""

    def __init__(self) -> None:
        """Initialize OpenAI LLM service."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing OpenAI client with model: {Config.OPENAI_MODEL_NAME}")

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model_name = Config.OPENAI_MODEL_NAME

        self.logger.info("OpenAI client initialized successfully")

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response using OpenAI API."""
        self.logger.info("Generating response with OpenAI...")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )

        return response.choices[0].message.content or ""


class AnthropicLLMService(BaseLLMService):
    """LLM service using Anthropic Claude API."""

    def __init__(self) -> None:
        """Initialize Anthropic LLM service."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing Anthropic client with model: {Config.ANTHROPIC_MODEL_NAME}")

        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model_name = Config.ANTHROPIC_MODEL_NAME

        self.logger.info("Anthropic client initialized successfully")

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response using Anthropic API."""
        self.logger.info("Generating response with Anthropic Claude...")

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        content_block = response.content[0]
        if hasattr(content_block, "text"):
            return content_block.text  # type: ignore
        return ""


class GeminiLLMService(BaseLLMService):
    """LLM service using Google Gemini API."""

    def __init__(self) -> None:
        """Initialize Gemini LLM service."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing Gemini client with model: {Config.GEMINI_MODEL_NAME}")

        genai.configure(api_key=Config.GEMINI_API_KEY)  # type: ignore
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)  # type: ignore

        self.logger.info("Gemini client initialized successfully")

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response using Gemini API."""
        self.logger.info("Generating response with Gemini...")

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(  # type: ignore
                max_output_tokens=max_tokens,
                temperature=0,
            ),
        )

        return response.text


def get_llm_service(provider: str) -> BaseLLMService:
    """
    Factory function to get the appropriate LLM service based on the selected provider.

    Args:
        provider: LLM provider to use (e.g., 'openai', 'anthropic', 'gemini')

    Returns:
        Initialized LLM service instance
    """
    llm_provider = LLMProvider(provider)

    logger.info(f"Creating LLM service for provider: {llm_provider.value}")

    if llm_provider == LLMProvider.OPENAI:
        return OpenAILLMService()
    elif llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicLLMService()
    elif llm_provider == LLMProvider.GEMINI:
        return GeminiLLMService()
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
