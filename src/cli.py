import os
import json
import asyncio
from dataclasses import dataclass
from typing import List

from .agent import run_agent


DEFAULT_NAV_PROMPTS = [
    "Scrape YC news for agentic tools",
    "Check TechCrunch for AI browser funding",
    "Look for recent funding on Crunchbase for AI browsers",
]


@dataclass
class CLIConfig:
    query: str
    agents: int


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment before running."
        )


async def run_all(config: CLIConfig) -> List[dict]:
    nav_prompts = DEFAULT_NAV_PROMPTS[: config.agents]
    if len(nav_prompts) < config.agents:
        # pad generic prompts
        for i in range(len(nav_prompts), config.agents):
            nav_prompts.append(f"General web scan for: {config.query}")

    tasks = []
    for idx, prompt in enumerate(nav_prompts):
        tasks.append(run_agent(agent_name=f"agent-{idx+1}", nav_prompt=prompt, query=config.query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for res in results:
        if isinstance(res, Exception):
            output.append({
                "agent_name": "unknown",
                "error": str(res),
            })
        else:
            output.append(res.dict())
    return output


def write_markdown(results: List[dict], query: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", "research_summary.md")
    lines = [f"# Research Summary\n", f"Query: {query}\n\n"]
    for r in results:
        name = r.get("agent_name", "agent")
        lines.append(f"## {name}\n")
        lines.append(f"Prompt: {r.get('nav_prompt','')}\n\n")
        lines.append("### Summary\n")
        lines.append(r.get("summary", "") + "\n\n")
        sources = r.get("sources", [])
        if sources:
            lines.append("### Sources\n")
            for s in sources[:10]:
                lines.append(f"- {s}\n")
            lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM whisperers: Playwright + LLM agents")
    parser.add_argument("--query", type=str, default="Research latest AI browser funding")
    parser.add_argument("--agents", type=int, default=4)
    args = parser.parse_args()

    _ensure_api_key()
    config = CLIConfig(query=args.query, agents=max(3, min(args.agents, 5)))

    results: List[dict] = asyncio.run(run_all(config))

    # Terminal JSON output
    print(json.dumps(results, indent=2))

    # Write markdown summary
    md_path = write_markdown(results, query=config.query)
    print(f"\nMarkdown summary written to: {md_path}")


if __name__ == "__main__":
    main()