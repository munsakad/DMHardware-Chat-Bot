"""
gemini_client.py
----------------
Thin wrapper around Google's Gemini API (google-genai SDK) that builds a
grounded, RAG-style prompt from retrieved knowledge base context and a
running chat history, then calls the model.
"""

import os

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # swap to any model your API key has access to

SYSTEM_INSTRUCTION = """You are DMHardware AI Chatbot, a domain expert assistant for \
Electronics, Robotics, PCB Design, and Digital Logic Design.

Ground your answers primarily in the CONTEXT provided below, which comes from a \
curated hardware-engineering knowledge base. If the context doesn't fully cover \
the question, you may use your own general engineering knowledge to fill gaps, \
but say so explicitly when you go beyond the provided context.

Be precise, concise, and practical — this tool is used by someone doing real \
electronics/robotics/PCB/digital-logic work. Use component names, units, and \
correct terminology. If a question is ambiguous, ask a brief clarifying question."""


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file or your shell "
            "environment before running the app."
        )
    return genai.Client(api_key=api_key)


def build_prompt(context: str, chat_history: list, user_message: str) -> str:
    history_text = ""
    for turn in chat_history[-6:]:  # keep last few turns for lightweight memory
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    return f"""CONTEXT (retrieved from knowledge base):
{context}

CONVERSATION SO FAR:
{history_text}

User's new question: {user_message}

Answer the user's question now, grounded in the CONTEXT above where relevant."""


def generate_response(
    client: genai.Client,
    context: str,
    chat_history: list,
    user_message: str,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_prompt(context, chat_history, user_message)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
        ),
    )
    return response.text
