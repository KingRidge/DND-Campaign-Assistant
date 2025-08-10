import os
import sys
import re
import json
import requests
from typing import Optional

# Ollama settings (local only)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MODEL", "qwen3:8b")  # Default model, can be overridden
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_KEY = os.getenv("OLLAMA_KEY", "")  # Optional API key for Ollama

def llm_chat_ollama(
    messages,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    expect_json: bool = False,
) -> str:

    model_name, server, key = model
    url = f"{server}/api/chat"
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }

    if expect_json:
        # Add JSON-specific prompt elements
        messages = [
            {
                "role": "system",
                "content": "You are a JSON generator. Your response must be a single, complete, valid JSON object. No other text.",
            },
            *messages,
        ]
        options["format"] = "json"  # Tell Ollama we want JSON
        options["stop"] = ["\n\n", "</s>", "```"]  # Stop markers

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    print(f"Sending request to Ollama...", file=sys.__stderr__)
    print(f"Model: {payload['model']}", file=sys.__stderr__)
    print(f"Temperature: {temperature}", file=sys.__stderr__)
    print(f"Expecting JSON: {expect_json}", file=sys.__stderr__)

    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT, headers=headers)
    r.raise_for_status()
    data = r.json()

    content = data.get("message", {}).get("content", "").strip()

    if not content:
        print("Received empty response from Ollama", file=sys.__stderr__)
        raise ValueError("Empty response from LLM")

    print(f"Raw response from Ollama:", file=sys.__stderr__)
    print("-" * 40, file=sys.__stderr__)
    print(content[:500], file=sys.__stderr__)
    print("-" * 40, file=sys.__stderr__)

    if expect_json:
        # Try to find a JSON object in the response
        try:
            # First, try to parse the whole response as JSON
            json.loads(content)
            return content
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the response
            try:
                # Look for content between triple backticks
                matches = re.findall(r"```(?:json)?\s*(.+?)\s*```", content, re.DOTALL)
                if matches:
                    content = matches[0]

                # Find the first { and last }
                start = content.find("{")
                end = content.rfind("}")

                if start >= 0 and end > start:
                    content = content[start : end + 1]
                    # Validate that this is valid JSON
                    json.loads(content)  # This will raise JSONDecodeError if invalid
                    return content

                raise ValueError("No JSON object found in response")
            except Exception as e:
                print(f"Error extracting JSON: {e}", file=sys.__stderr__)
                print("Attempted to parse:", file=sys.__stderr__)
                print(content, file=sys.__stderr__)
                raise

    return content
