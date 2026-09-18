# Session 9 pre-reading

Complete this short preparation before 28 September 2026.

## Required (about 25 minutes)

1. Read the sections **Building blocks, workflows, and agents** and **When (and when not) to use agents** in [Anthropic's Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).
2. Read the [OpenAI Agents SDK overview](https://openai.github.io/openai-agents-python/) and note the roles of an agent, a runner, a function tool, and a handoff.
3. Skim [What's new in MCP Python SDK v2](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/whats-new.md), focusing on the `FastMCP` to `MCPServer` rename.

## Bring to class

- A working Python 3.12 `.venv` with `requirements.txt` installed.
- An OpenAI API key available through a local `.env` file; never paste it into a notebook.
- One example of a task that should remain a deterministic workflow and one that could benefit from an agent.

## Reflection prompt

For your agent example, identify one tool it needs, one failure mode, and one action that should require human approval.

