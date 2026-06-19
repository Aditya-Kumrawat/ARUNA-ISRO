from __future__ import annotations
from aruna.l7_llm.client import BriefingLLM, AzureLangChainClient, MockLLMClient


class LLMFactory:
    @staticmethod
    def from_config(config: dict | None = None) -> BriefingLLM:
        cfg = (config or {}).get("llm", {})
        provider = cfg.get("provider", "mock")
        model = cfg.get("model", "gpt-4.1-mini")
        api_version = cfg.get("api_version", "2024-02-01")
        temperature = float(cfg.get("temperature", 0.1))
        max_tokens = int(cfg.get("max_tokens", 400))

        if provider == "azure":
            return AzureLangChainClient(
                model=model,
                api_version=api_version,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider == "ollama":
            raise NotImplementedError("Ollama provider not yet implemented")
        else:
            return MockLLMClient()
