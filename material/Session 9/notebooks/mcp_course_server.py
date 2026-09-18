"""
ANLP Course MCP Server
======================
A minimal MCP server exposing two tools:
  - get_session_topics(session_number) → topics for that session
  - search_glossary(term)              → definition of a course term

Run it standalone:
    python mcp_course_server.py

Or point any MCP client (Claude Desktop, Cursor, VS Code) at it:
    {
      "mcpServers": {
        "anlp": {
          "command": "python",
          "args": ["path/to/mcp_course_server.py"]
        }
      }
    }

Used by notebook 01_agents_concepts.ipynb (Section 4: MCP) to demonstrate
an agent auto-discovering and calling tools via the MCP protocol.
"""

from mcp.server import MCPServer

mcp_server = MCPServer("ANLP Course Server")


@mcp_server.tool()
def get_session_topics(session_number: int) -> str:
    """Get the topics covered in a given ANLP course session."""
    topics = {
        7: "LLM Deep Dive, Prompting Techniques (CoT, few-shot), LLM Evaluation (BLEU, ROUGE, LLM-as-judge)",
        8: "LLM API Access, RAG, FAISS, Streaming, Gradio/Streamlit, Tool Use",
        9: "AI Agents, Workflow Patterns, MCP, OpenAI Agents SDK",
    }
    return topics.get(session_number, f"Session {session_number} not in course")


@mcp_server.tool()
def search_glossary(term: str) -> str:
    """Look up a term in the ANLP course glossary."""
    glossary = {
        "rag": "Retrieval-Augmented Generation — grounding LLM answers in retrieved documents",
        "embedding": "A dense numerical vector representing text meaning",
        "faiss": "Facebook AI Similarity Search — fast vector similarity search library",
        "mcp": "Model Context Protocol — open standard for connecting AI models to tools",
        "agent": "An LLM extended with tools that can take multi-step actions toward a goal",
        "bm25": "A lexical search algorithm scoring by term frequency and document rarity",
        "cosine similarity": "A measure of angle between two vectors, bounded between -1 and 1",
    }
    return glossary.get(term.lower(), f"'{term}' not in glossary — try a related term")


if __name__ == "__main__":
    mcp_server.run()
