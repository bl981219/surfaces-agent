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
from surfaces_agent.tools.analysis import extract_vasp_characteristics
from surfaces_agent.tools.md import run_chgnet_md
from surfaces_agent.tools.supercell import make_supercell

def main():
    # Load .env first to ensure all API keys are available
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the Surfaces Agent.")
    parser.add_argument("--model", type=str, help="Override AGENT_MODEL from .env")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    model_id = args.model or os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite-preview")

    if not api_key:
        print("Error: 'API_KEY' environment variable is not set. Please check your .env file.")
        sys.exit(1)

    # Setup Logging
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    session_log = output_dir / "agent_session.log"
    
    def log_interaction(prompt, response):
        with open(session_log, "a") as f:
            f.write(f"\n[USER] >> {prompt}\n")
            f.write(f"[AGENT] {response}\n")

    client = genai.Client(api_key=api_key)
    print(f"🤖 surfaces-agent initialized with {model_id}. Type 'exit' to quit.")
    print(f"📄 Session log: {session_log}")

    agent_tools = [
        fetch_bulk_structure,
        generate_and_relax_slab,
        generate_adsorption_configs,
        save_structure,
        search_scientific_knowledge,
        extract_vasp_characteristics,
        run_chgnet_md,
        make_supercell
    ]

    system_instruction = (
        "You are an autonomous computational materials science orchestrator. "
        "CRITICAL: If the user provides a file path (e.g., 'output/SrTiO3.cif'), you MUST use that file directly "
        "with structural tools. DO NOT call 'fetch_bulk_structure' if a local file is available or specified. "
        "Use search_scientific_knowledge for literature benchmarks and DOIs. "
        "Use structural tools to generate slabs, configurations, supercells, or run MD simulations. "
        "Use extract_vasp_characteristics to read and analyze completed VASP output files."
    )

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=agent_tools,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
        )
    )

    while True:
        try:
            prompt = input("\n>> ")
            if not prompt:
                continue
            if prompt.strip().lower() in ['exit', 'quit']:
                break
            
            print("🤖 Processing...")
            response = chat.send_message(prompt)
            
            text_response = response.text or "[Agent completed task with tool calls]"
            print(f"\n{text_response}")
            
            # Log the session
            log_interaction(prompt, text_response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            error_msg = f"Agent engine failed: {e}"
            print(error_msg)
            log_interaction("SYSTEM_ERROR", error_msg)

if __name__ == "__main__":
    main()