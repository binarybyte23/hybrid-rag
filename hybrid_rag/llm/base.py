class LLMProvider:
    async def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError
