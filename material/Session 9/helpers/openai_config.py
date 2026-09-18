"""Consistent OpenAI configuration for Session 9 examples."""

from __future__ import annotations

import os

from agents import set_default_openai_client
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


DEFAULT_TEACHING_MODEL = "gpt-5.6-luna"


def openai_client_options() -> dict[str, str]:
    """Return OpenAI client options without exposing credential values."""
    load_dotenv()
    options = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
    if organization := os.environ.get("OPENAI_ORG_ID"):
        options["organization"] = organization
    if project := os.environ.get("OPENAI_PROJECT_ID"):
        options["project"] = project
    return options


def require_openai_client() -> OpenAI:
    """Create a configured synchronous client for embeddings and responses."""
    options = openai_client_options()
    if not options["api_key"]:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the repository .env file before running live examples."
        )
    return OpenAI(**options)


def configure_agents_sdk() -> None:
    """Configure the Agents SDK with the same account scope as the OpenAI client."""
    options = openai_client_options()
    if not options["api_key"]:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the repository .env file before running live examples."
        )
    set_default_openai_client(AsyncOpenAI(**options))


def teaching_model() -> str:
    """Return the course model, allowing instructors to override it in `.env`."""
    return os.environ.get("OPENAI_MODEL", DEFAULT_TEACHING_MODEL)
