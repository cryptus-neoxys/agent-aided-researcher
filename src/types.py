from typing import List, Optional, Literal

from langchain_core.pydantic_v1 import BaseModel, Field


class Action(BaseModel):
    action: Literal["goto", "search", "click", "extract", "scroll", "wait", "end"]
    url: Optional[str] = Field(default=None, description="Target URL for navigation")
    selector: Optional[str] = Field(default=None, description="CSS selector for element")
    query: Optional[str] = Field(default=None, description="Search query text")
    text: Optional[str] = Field(default=None, description="Text to fill or use")
    notes: Optional[str] = Field(default=None, description="Extra notes or constraints")


class ActionPlan(BaseModel):
    steps: List[Action] = Field(default_factory=list, description="Ordered list of actions")


class AgentResult(BaseModel):
    agent_name: str
    nav_prompt: str
    steps_executed: int
    summary: str
    sources: List[str]
    errors: List[str]
    raw_extracts: List[str]