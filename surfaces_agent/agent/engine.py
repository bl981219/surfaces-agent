# surfaces_agent/agent/engine.py
import argparse
import sys
from surfaces_agent.agent.registry import ToolRegistry
from surfaces_agent.agent.state import ExecutionState
from surfaces_agent.llm.client import GeminiClient
from surfaces_agent.tools.mp import MPQuerySchema, fetch_bulk_structure
from surfaces_agent.tools.io import SaveStructureSchema, save_structure
from surfaces_agent.tools.slab import SlabRelaxationSchema, generate_and_relax_slab

# Import your scientific tools and schemas
from surfaces_agent.tools.mp import MPQuerySchema, fetch_bulk_structure

def build_registry(state: ExecutionState) -> ToolRegistry:
    registry = ToolRegistry(state)
    
    # Materials Project Query Tool
    registry.register(
        name="fetch_bulk_structure",
        description="Fetches the lowest energy bulk structure for a given chemical formula from the Materials Project database.",
        schema=MPQuerySchema,
        func=fetch_bulk_structure
    )
    
    # Save Structure Tool
    registry.register(
        name="save_structure",
        description="Saves a structure from the execution state to a local file. Only use this if the user explicitly asks to save or export a file.",
        schema=SaveStructureSchema,
        func=save_structure
    )

    # Surface Cleave & MLFF Relaxation Tool
    registry.register(
        name="generate_and_relax_slab",
        description="Cleaves a surface slab from a bulk structure reference ID, relaxes it using the CHGNet ML force field, and calculates the surface energy.",
        schema=SlabRelaxationSchema,
        func=generate_and_relax_slab
    )

    # Future tools will go here:
    # registry.register(name="generate_slab", ...)
    
    return registry

def run_agent_loop(initial_prompt: str, model: str, temperature: float, max_steps: int = 5):
    """The core ReAct orchestration loop with interactive shell support."""
    state = ExecutionState()
    registry = build_registry(state)
    llm_tools = registry.get_llm_tools()
    client = GeminiClient(model_name=model, temperature=temperature)
    
    # This history array now persists for the entire session
    history = [] 
    
    # Strong system instruction to prevent lazy stopping
    system_instruction = (
        "You are an autonomous scientific orchestrator. "
        "You must execute ALL steps requested by the user before returning a final answer. "
        "Do not stop halfway to ask for permission if the user has already given you a multi-step command."
    )
    
    print("🤖 surfaces-agent interactive shell initialized. Type 'exit' to quit.\n")
    
    current_prompt = f"{system_instruction}\n\nUser Request: {initial_prompt}" if initial_prompt else None

    while True:
        # If we don't have a prompt (e.g., starting empty or finished previous task), ask the user
        if not current_prompt:
            try:
                user_input = input("\n>> ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting surfaces-agent. State cleared.")
                    break
                current_prompt = user_input
            except (KeyboardInterrupt, EOFError):
                print("\nExiting surfaces-agent. State cleared.")
                break

        print(f"\n🤖 Processing...")
        
        # Tool execution loop for the current prompt
        for step in range(max_steps):
            response = client.generate_with_tools(current_prompt, llm_tools, history)
            
            if response["action"] == "reply":
                print(f"✅ Final Answer:\n{response['text']}")
                # Add the final interaction to history so it remembers the context
                history.append({"role": "user", "content": current_prompt})
                history.append({"role": "assistant", "content": response['text']})
                
                # Clear current prompt to wait for next user input
                current_prompt = None 
                break

            elif response["action"] == "call_tool":
                tool_name = response["tool_name"]
                tool_args = response["tool_args"]
                print(f"   🔧 Routing to tool: {tool_name} | Args: {tool_args}")
                
                observation = registry.execute(tool_name, tool_args)
                print(f"   📊 Tool Output: {observation}")
                
                current_prompt = (
                    f"Observation from {tool_name}: {observation}\n"
                    f"Evaluate if the original user request is fully complete. "
                    f"If there are remaining steps, execute the next tool. "
                    f"If complete, summarize the results."
                )
                # Keep track of tool usage in history so the LLM remembers what it did
                history.append({"tool": tool_name, "output": observation})
                
        if current_prompt is not None:
             print("\n⚠️ Agent reached maximum steps without a final answer. Returning to prompt.")
             current_prompt = None

def main():
    parser = argparse.ArgumentParser(description="Run the surfaces-agent orchestrator.")
    # Make prompt optional so we can boot directly into the shell
    parser.add_argument("--prompt", type=str, default=None, help="Initial scientific query")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="LLM provider model string")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum number of iterations per query")
    
    args = parser.parse_args()
    
    import sys
    try:
        run_agent_loop(args.prompt, args.model, args.temperature, args.max_steps)
    except Exception as e:
        print(f"Agent engine failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()