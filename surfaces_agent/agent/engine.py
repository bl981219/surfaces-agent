# surfaces_agent/agent/engine.py
import argparse
import sys
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Tool imports
from surfaces_agent.tools.mp import fetch_bulk_structure
from surfaces_agent.tools.slab import generate_and_relax_slab
from surfaces_agent.tools.adsorption import generate_adsorption_configs
from surfaces_agent.tools.io import save_structure
from surfaces_agent.tools.search import search_scientific_knowledge

def main():
    parser = argparse.ArgumentParser(description="Run the Surfaces Agent.")
    parser.add_argument("--model", type=str, help="Override AGENT_MODEL from .env")
    args = parser.parse_args()

    load_dotenv()
    
    # Priority: CLI flag > .env (AGENT_MODEL) > Default
    api_key = os.environ.get("API_KEY")
    model_id = args.model or os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite-preview")

    client = genai.Client(api_key=api_key)
    print(f"🤖 surfaces-agent initialized with {model_id}. Type 'exit' to quit.")

    agent_tools = [
        search_scientific_knowledge,
        fetch_bulk_structure,
        generate_and_relax_slab,
        generate_adsorption_configs,
        save_structure
    ]

    system_instruction = (
        "You are an autonomous computational materials science orchestrator. "
        "Use search_scientific_knowledge for literature benchmarks and DOIs. "
        "Use structural tools only when a calculation or file generation is requested."
    )

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=agent_tools,
            temperature=0.1,
        )
    )

    while True:
        try:
            prompt = input("\n>> ")
            if prompt.strip().lower() in ['exit', 'quit']:
                break
            
            print("🤖 Processing...")
            response = chat.send_message(prompt)
            print(f"\n{response.text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Agent engine failed: {e}")

if __name__ == "__main__":
    main()