# surfaces_agent/agent/engine.py
import argparse
import sys
from surfaces_agent.agent.registry import ToolRegistry
from surfaces_agent.agent.state import ExecutionState
from surfaces_agent.llm.client import GeminiClient

# Import your scientific tools and schemas
from surfaces_agent.tools.mp import MPQuerySchema, fetch_bulk_structure

def build_registry(state: ExecutionState) -> ToolRegistry:
    registry = ToolRegistry(state)
    
    # Register the Materials Project Tool
    registry.register(
        name="fetch_bulk_structure",
        description="Fetches the lowest energy bulk structure for a given chemical formula from the Materials Project database.",
        schema=MPQuerySchema,
        func=fetch_bulk_structure
    )
    
    # Future tools will go here:
    # registry.register(name="generate_slab", ...)
    
    return registry

def run_agent_loop(prompt: str, model: str, temperature: float, max_steps: int = 5):
    """The core ReAct (Reason + Act) orchestration loop."""
    
    # 1. Initialize the shared state blackboard
    state = ExecutionState()
    
    # 2. Build the registry with the shared state
    registry = build_registry(state)
    llm_tools = registry.get_llm_tools()
    client = GeminiClient(model_name=model, temperature=temperature)
    
    current_prompt = prompt
    history = [] 
    
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
            current_prompt = f"Observation from {tool_name}: {observation}\nWhat is the next step? If the task is complete, summarize the results."
            history.append({"tool": tool_name, "output": observation})

    print("\n⚠️ Agent reached maximum steps without a final answer.")

def main():
    parser = argparse.ArgumentParser(description="Run the surfaces-agent orchestrator loop.")
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="The scientific query (e.g., 'Fetch the bulk structure of SrTiO3 from Materials Project.')"
    )
    # Defaulting to flash to save your API quota
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="LLM provider model string")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of iterations")
    
    args = parser.parse_args()
    
    try:
        run_agent_loop(args.prompt, args.model, args.temperature, args.max_steps)
    except Exception as e:
        print(f"Agent engine failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()