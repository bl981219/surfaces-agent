# surfaces_agent/agent/engine.py
import argparse
import sys
from surfaces_agent.agent.registry import ToolRegistry
from surfaces_agent.llm.client import GeminiClient

# Import your scientific tools (assuming they exist in the tools directory)
# from surfaces_agent.tools.slab import SlabParameters, generate_slab

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    # Example registration (uncomment when slab.py is fully implemented):
    # registry.register(
    #     name="generate_slab",
    #     description="Generates a surface slab from a bulk formula and Miller indices.",
    #     schema=SlabParameters,
    #     func=generate_slab
    # )
    return registry

def run_agent_loop(prompt: str, model: str, temperature: float, max_steps: int = 5):
    """The core ReAct (Reason + Act) orchestration loop."""
    registry = build_registry()
    llm_tools = registry.get_llm_tools()
    client = GeminiClient(model_name=model, temperature=temperature)
    
    current_prompt = prompt
    history = [] # Tracks the context of tool outputs
    
    print(f"🤖 Starting surfaces-agent loop for task:\n'{prompt}'\n")

    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")
        response = client.generate_with_tools(current_prompt, llm_tools, history)
        
        if response["action"] == "reply":
            print("\n✅ Final Answer:")
            print(response["text"])
            return

        elif response["action"] == "call_tool":
            tool_name = response["tool_name"]
            tool_args = response["tool_args"]
            print(f"🔧 LLM routing to tool: {tool_name}")
            print(f"   Parameters: {tool_args}")
            
            # Execute the deterministic physics code
            observation = registry.execute(tool_name, tool_args)
            print(f"📊 Tool Output: {observation}")
            
            # Feed the observation back into the prompt for the next loop iteration
            current_prompt = f"Observation from {tool_name}: {observation}\nWhat is the next step?"
            history.append({"tool": tool_name, "output": observation})

    print("\n⚠️ Agent reached maximum steps without a final answer.")

def main():
    parser = argparse.ArgumentParser(description="Run the surfaces-agent orchestrator loop.")
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="The scientific query (e.g., 'Analyze the O* intermediate adsorption on a doped SrTiO3 (001) surface.')"
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="LLM provider model string")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (0.0 recommended for orchestration)")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of tool-calling iterations")
    
    args = parser.parse_args()
    
    try:
        run_agent_loop(args.prompt, args.model, args.temperature, args.max_steps)
    except Exception as e:
        print(f"Agent engine failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()