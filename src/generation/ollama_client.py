from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class OllamaResponse:
    content: str
    model: str

    total_duration_ns: int
    load_duration_ns: int

    prompt_eval_count: int
    prompt_eval_duration_ns: int

    eval_count: int
    eval_duration_ns: int


class OllamaClient:

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> OllamaResponse:

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return OllamaResponse(
            content=data.get(
                "message",
                {},
            ).get(
                "content",
                "",
            ),
            model=data.get("model", model),

            total_duration_ns=data.get(
                "total_duration",
                0,
            ),
            load_duration_ns=data.get(
                "load_duration",
                0,
            ),

            prompt_eval_count=data.get(
                "prompt_eval_count",
                0,
            ),
            prompt_eval_duration_ns=data.get(
                "prompt_eval_duration",
                0,
            ),

            eval_count=data.get(
                "eval_count",
                0,
            ),
            eval_duration_ns=data.get(
                "eval_duration",
                0,
            ),
        )


def ns_to_ms(value: int) -> float:
    return value / 1_000_000
