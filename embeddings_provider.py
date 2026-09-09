"""Proveedor de embeddings basado en la API de OpenRouter (compatible con OpenAI).

Se usa una API remota en lugar de sentence-transformers/HuggingFace porque las
politicas de red de la maquina bloquean el handshake TLS contra huggingface.co,
lo que impide descargar los modelos locales.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / "secrets" / ".env"

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/text-embedding-3-small"


def build_embeddings(
    model_name: str = DEFAULT_MODEL,
    api_base: str = OPENROUTER_API_BASE,
) -> OpenAIEmbeddings:
    """Crea el cliente de embeddings apuntando a OpenRouter."""
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"No se encontro OPENAI_API_KEY en {ENV_FILE}")

    return OpenAIEmbeddings(
        model=model_name,
        base_url=api_base,
        api_key=api_key,
        # OpenRouter espera texto plano; sin esto LangChain envia arrays de
        # tokens y necesita descargar el vocabulario de tiktoken.
        check_embedding_ctx_length=False,
    )
