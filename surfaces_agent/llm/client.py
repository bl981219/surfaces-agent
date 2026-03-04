import os
from google import genai
from google.genai import types
from dotenv import load_dotenv  # <-- 1. Import this

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        load_dotenv()  # <-- 2. Call this before reading os.environ
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or .env entry not set.")
            
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def generate_with_tools(self, prompt: str, tools: list[dict], history: list = None) -> dict:
        function_declarations = [t["function"] for t in tools]
        
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            tools=[types.Tool(function_declarations=function_declarations)]
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        if response.function_calls:
            fc = response.function_calls[0]
            return {
                "action": "call_tool",
                "tool_name": fc.name,
                "tool_args": dict(fc.args) if fc.args else {}
            }
        else:
            return {
                "action": "reply",
                "text": response.text
            }