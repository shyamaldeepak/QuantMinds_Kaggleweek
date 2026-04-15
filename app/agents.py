import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag.client import get_openai_client
from scripts.rag import retrieve, generate_answer, load_index

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

# Load environment using same config as app
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# LLM Client setup
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths for RAG
index_path = str(project_root / "data" / "my_index.faiss")
chunks_path = str(project_root / "data" / "chunks.json")

class InternalResearcher:
    """Agent 1: Searches internal corpus and generates Output 1."""
    def __init__(self):
        # We load the index so we can query our local FAISS knowledge base
        self.index, self.chunks = load_index(index_path, chunks_path)

    async def execute(self, query: str) -> str:
        print("\n--- Agent 1 (Internal Researcher) ---")
        results = retrieve(query, self.index, self.chunks, k=4)
        if not results:
            print("No matching internal information found.")
            return "No matching internal information found."
        
        answer = generate_answer(query, results)
        print(f"-> Output 1 generated ({len(answer)} chars)")
        return answer

class ExternalFactChecker:
    """Agent 2: Validates with online info, databases via MCP server."""
    def __init__(self):
        # Configure MCP client to connect to our fastmcp server
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(project_root / "app" / "mcp_server.py")]
        )

    async def execute(self, query: str, internal_output: str) -> str:
        print("\n--- Agent 2 (External Fact Checker) ---")
        system_prompt = (
            "You are an External Fact Checker. You receive a user query and an internal document response. "
            "Use your tools (search_web, search_wikipedia) to verify the facts and fetch up-to-date information. "
            "If requested or useful, save a report using create_markdown_report. "
            "Return the validated fact-checked information. Focus on supplementing or correcting the internal output."
        )
        
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                
                # Format tools for OpenAI function calling
                openai_tools = []
                for t in tools_response.tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema
                        }
                    })

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nInternal Output:\n{internal_output}"}
                ]
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini", # Typically reliable enough for tool orchestration
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                
                msg = response.choices[0].message
                
                if msg.tool_calls:
                    messages.append(msg)
                    for tool_call in msg.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        print(f"-> Triggering MCP action: {tool_call.function.name} with {args}")
                        
                        tool_result = await session.call_tool(tool_call.function.name, args)
                        
                        # Parse MCP result content
                        content = tool_result.content[0].text if tool_result.content else str(tool_result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content
                        })
                    
                    # Final completion after tool results
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    msg = response.choices[0].message

                print(f"-> Output 2 generated ({len(msg.content)} chars)")
                return msg.content

class Synthesizer:
    """Agent 3: Combines all outputs to generate final answer."""
    async def execute(self, query: str, internal_output: str, external_output: str) -> str:
        print("\n--- Agent 3 (Synthesizer) ---")
        prompt = (
            f"User Query: {query}\n\n"
            f"Output 1 (Internal): {internal_output}\n\n"
            f"Output 2 (External validation): {external_output}\n\n"
            "Based on the query and the two outputs, produce a final, structured, comprehensive response. "
            "Explicitly highlight what was found in the internal codebase versus what was verified externally."
        )
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"-> Output 3 generated")
        return response.choices[0].message.content

class Visualizer:
    """Agent 4: Scans the final output for numeric comparisons and generates a graph if appropriate."""
    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(project_root / "app" / "mcp_server.py")]
        )

    async def execute(self, text_output: str) -> str:
        print("\n--- Agent 4 (Visualizer) ---")
        system_prompt = (
            "You are a Visualizer Agent. You read a final report and determine if there is numerical data "
            "that could be plotted (e.g. comparing 2024 vs 2025 revenue, or company comparisons). "
            "If yes, use the generate_graph tool to create a visual chart, and reply with ONLY a simple message "
            "alerting the user that the graph was created. If there is no good data to plot, reply exactly with 'SKIP'."
        )
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                
                openai_tools = []
                for t in tools_response.tools:
                    if t.name == "generate_graph":
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.inputSchema
                            }
                        })
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_output}
                ]
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                
                msg = response.choices[0].message
                
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        print(f"-> Generating visual graph using MCP: {args}")
                        tool_result = await session.call_tool(tool_call.function.name, args)
                        content = tool_result.content[0].text if tool_result.content else str(tool_result)
                        return f"\n\n[Visualization generated]: {content}"
                
                return ""

class Supervisor:
    """Orchestrator that decides flow and retries"""
    def __init__(self):
        self.agent1 = InternalResearcher()
        self.agent2 = ExternalFactChecker()
        self.agent3 = Synthesizer()
        self.agent4 = Visualizer()

    async def run(self, query: str):
        print(f"\n=======================================================\nSupervisor: Starting flow for query: '{query}'\n=======================================================")
        
        # Step 1
        out1 = await self.agent1.execute(query)
        
        # Basic feedback logic: if internal researcher failed to find much, inform Agent 2 directly.
        if "No matching internal information" in out1:
            out1 += "\n(Supervisor Note: Internal search failed. Rely entirely on external validation.)"
            
        # Step 2
        out2 = await self.agent2.execute(query, out1)
        
        # Step 3
        final_out = await self.agent3.execute(query, out1, out2)
        
        # Step 4: Optional Visualization
        visual_out = await self.agent4.execute(final_out)
        if visual_out and "SKIP" not in visual_out:
            final_out += visual_out
        
        print("\n=======================================================\nSupervisor: Flow Complete. Final Answer:\n=======================================================\n")
        print(final_out)
        return final_out

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the latest news and stock price of Apple compared to my internal report?"
    supervisor = Supervisor()
    asyncio.run(supervisor.run(query))
