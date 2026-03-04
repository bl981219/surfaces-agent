import argparse
import os
import sys
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent_executor import AgentExecutor
from surfaces_agent.tools import (
    retrieve_surface_literature, 
    fetch_bulk_structure,
    generate_slab_with_dynamics, 
    relax_and_calculate_surface_energy
)

def main():
    parser = argparse.ArgumentParser(
        description="surfaces-agent: AI-driven workflow for surface generation and MLIAP relaxation."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="Natural language instruction for the agent."
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        choices=["gemini", "openai"], 
        default="gemini", 
        help="Choose the LLM backend. Defaults to gemini."
    )
    args = parser.parse_args()

    # Initialize the LLM based on user selection
    if args.llm == "gemini":
        if "GOOGLE_API_KEY" not in os.environ:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            sys.exit(1)
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Gemini 1.5 Pro is highly recommended for complex tool calling
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0) 
    else:
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Load all four tools
    tools = [
        retrieve_surface_literature, 
        fetch_bulk_structure,
        generate_slab_with_dynamics, 
        relax_and_calculate_surface_energy
    ]

    # The System Prompt now explicitly handles the polymorph fallback
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert computational materials scientist AI. Use your tools to fetch literature, download bulk structures from the Materials Project, generate slabs with selective dynamics, and execute MLIAP relaxations. If `fetch_bulk_structure` returns multiple polymorphs, you MUST stop and ask the user which space group to use before continuing."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    
    # verbose=True allows you to see the agent "thinking" in the terminal
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print(f"\n[surfaces-agent] Backend: {args.llm.upper()}")
    print(f"[surfaces-agent] Executing workflow for: '{args.prompt}'\n")
    
    try:
        result = agent_executor.invoke({"input": args.prompt})
        print("\n--- Final Agent Report ---")
        print(result["output"])
    except Exception as e:
        print(f"\n[Error]: Agent execution failed: {str(e)}")

if __name__ == "__main__":
    main()