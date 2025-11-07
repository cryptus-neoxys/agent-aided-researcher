import os
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from .types import ActionPlan


SYSTEM_PROMPT = (
    "You are a disciplined web agent operating in a browser. "
    "Given an instruction, break it into concrete steps using only these actions: "
    "goto, search, click, extract, scroll, wait, end. "
    "Rules:\n"
    "- Prefer duckduckgo.com for web search.\n"
    "- Provide CSS selectors for click/extract when possible.\n"
    "- Keep steps to 3-6 max.\n"
    "- Always finish with an 'end' action.\n"
)

USER_PROMPT = (
    "You are a web agent. Given instruction: {user_input}, "
    "break it into steps: navigate, extract, act. "
    "Output strictly as a JSON object with schema: {{\n"
    "  \"steps\": [{{\n"
    "    \"action\": one of ['goto','search','click','extract','scroll','wait','end'],\n"
    "    \"url\": optional string,\n"
    "    \"selector\": optional string,\n"
    "    \"query\": optional string,\n"
    "    \"text\": optional string,\n"
    "    \"notes\": optional string\n"
    "  }}]\n"
    "}}"
)


def _resolve_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class LLMPlanner:
    def __init__(self, temperature: float = 0.2, model: Optional[str] = None):
        self.model_name = model or _resolve_model()
        self.llm = ChatOpenAI(model=self.model_name, temperature=temperature)
        self.prompt: Runnable = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])

        # Structured output parser
        self.structured = self.llm.with_structured_output(ActionPlan)

    async def plan_actions(self, instruction: str) -> ActionPlan:
        chain = self.prompt | self.structured
        try:
            return await chain.ainvoke({"user_input": instruction})
        except Exception:
            # Fallback: try non-structured and lenient parse
            txt = await (self.prompt | self.llm).ainvoke({"user_input": instruction})
            content = getattr(txt, "content", "{}")
            # Naive JSON extraction
            import json, re
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return ActionPlan(**data)
                except Exception:
                    pass
            # Very conservative default plan
            return ActionPlan(steps=[])

    async def summarize(self, notes: str) -> str:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You write concise, source-linked research summaries."),
            ("human", "Summarize these extracted notes into 4-6 bullets with short source refs.\n\n{notes}"),
        ])
        txt = await (summary_prompt | self.llm).ainvoke({"notes": notes})
        return getattr(txt, "content", "")