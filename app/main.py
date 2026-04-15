import asyncio
from app.agents import Supervisor

async def display_hitl(query: str):
    print("\n" + "="*60)
    print("🤖 Welcome to QuantMinds Multi-Agent + Human-in-the-Loop!")
    print("="*60)
    print(f"Target Query: '{query}'\n")

    # Initialize orchestrator
    supervisor = Supervisor()
    
    # ----------------------------------------------------
    # Step 1: Internal Search
    # ----------------------------------------------------
    print("[SYSTEM] Starting Agent 1 (Internal Researcher)...")
    out1 = await supervisor.agent1.execute(query)
    
    print("\n" + "-"*40)
    print("[Agent 1 Output]")
    print(out1)
    print("-"*40)
    
    # HITL Check 1
    feedback1 = input("\n[HUMAN-IN-THE-LOOP] Does this internal information look complete? \n(Type 'y' to accept and generate final answer, 'n' to trigger Agent 2 for external verification, or specify a follow-up directive): ")
    
    out2 = "Skipped by Human"
    if feedback1.lower() != 'y':
        # Feedback injected into context if it's text, otherwise simple trigger
        instruction = f"User feedback: {feedback1}" if feedback1.lower() != 'n' else ""
        
        # ----------------------------------------------------
        # Step 2: External Search / MCP Tool Use
        # ----------------------------------------------------
        print("\n[SYSTEM] Agent 1 information was insufficient. Starting Agent 2 (External Fact Checker) via MCP Server...")
        out2 = await supervisor.agent2.execute(query + "\n" + instruction, out1)
        
        print("\n" + "-"*40)
        print("[Agent 2 Output]")
        print(out2)
        print("-"*40)
        
        # HITL Check 2
        feedback2 = input("\n[HUMAN-IN-THE-LOOP] Do you approve Agent 2's validation? (y/n): ")
        if feedback2.lower() != 'y':
            print("\n[SYSTEM] Process aborted by Human.")
            return

    # ----------------------------------------------------
    # Step 3: Final Synthesis
    # ----------------------------------------------------
    print("\n[SYSTEM] Generating final synthesized report using Agent 3...")
    final_out = await supervisor.agent3.execute(query, out1, out2)
    
    print("\n" + "="*60)
    print("✅ FINAL SYNTHESIZED RESPONSE")
    print("="*60)
    print(final_out)
    print("="*60)


if __name__ == "__main__":
    import sys
    default_query = "What is the latest strategic direction of Apple according to our internal data, and what is the latest news online?"
    user_query = sys.argv[1] if len(sys.argv) > 1 else input(f"Enter your query (or press Enter to use default: '{default_query}'):\n> ")
    if not user_query.strip():
        user_query = default_query
        
    asyncio.run(display_hitl(user_query))
