import asyncio
from typing import List, Tuple

from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeoutError

from .types import AgentResult, Action, ActionPlan
from .llm_planner import LLMPlanner


DEFAULT_TIMEOUT_MS = 15000


async def _safe_extract(page: Page, selector: str | None) -> str:
    try:
        if selector:
            await page.wait_for_selector(selector, timeout=DEFAULT_TIMEOUT_MS)
            el = await page.query_selector(selector)
            if el:
                txt = await el.inner_text()
                if txt:
                    return txt.strip()
        # fallback: grab main content
        candidates = ["article", "main", "div#content", "section", "body"]
        for sel in candidates:
            el = await page.query_selector(sel)
            if el:
                txt = await el.inner_text()
                if txt:
                    return txt.strip()
    except PlaywrightTimeoutError:
        return "[timeout extracting content]"
    except Exception as e:
        return f"[error extracting content: {e}]"
    return ""


async def _perform(page: Page, action: Action) -> Tuple[str, str | None]:
    source_url = None
    try:
        if action.action == "goto" and action.url:
            await page.goto(action.url, timeout=DEFAULT_TIMEOUT_MS)
            source_url = page.url
            return (f"Visited {page.url}", source_url)
        elif action.action == "search":
            q = action.query or action.text or ""
            # Prefer DuckDuckGo for simplicity
            await page.goto("https://duckduckgo.com/", timeout=DEFAULT_TIMEOUT_MS)
            await page.fill("input[name='q']", q)
            await page.keyboard.press("Enter")
            await page.wait_for_selector("#links", timeout=DEFAULT_TIMEOUT_MS)
            # click first result
            first_link = await page.query_selector("#links .result__title a, #links a.result__a")
            if first_link:
                await first_link.click()
                await page.wait_for_load_state("domcontentloaded")
            source_url = page.url
            return (f"Searched and opened: {q} -> {page.url}", source_url)
        elif action.action == "click" and action.selector:
            await page.wait_for_selector(action.selector, timeout=DEFAULT_TIMEOUT_MS)
            await page.click(action.selector)
            await page.wait_for_load_state("domcontentloaded")
            source_url = page.url
            return (f"Clicked {action.selector}", source_url)
        elif action.action == "extract":
            txt = await _safe_extract(page, action.selector)
            source_url = page.url
            return (txt, source_url)
        elif action.action == "scroll":
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            source_url = page.url
            return ("Scrolled page", source_url)
        elif action.action == "wait":
            await asyncio.sleep(1.0)
            source_url = page.url
            return ("Waited briefly", source_url)
        elif action.action == "end":
            return ("[end]", page.url)
    except PlaywrightTimeoutError:
        return (f"[timeout on action {action.action}]", page.url)
    except Exception as e:
        return (f"[error on action {action.action}: {e}]", page.url)
    return ("", page.url)


async def run_agent(agent_name: str, nav_prompt: str, query: str) -> AgentResult:
    planner = LLMPlanner()
    instruction = f"{nav_prompt}. Core query: {query}"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)

        raw_extracts: List[str] = []
        sources: List[str] = []
        errors: List[str] = []

        plan: ActionPlan = await planner.plan_actions(instruction)
        if not plan.steps:
            # provide a basic plan if LLM failed
            plan = ActionPlan(steps=[
                Action(action="search", query=query),
                Action(action="extract", selector="article, main"),
                Action(action="end"),
            ])

        steps_executed = 0
        for step in plan.steps:
            outcome, src = await _perform(page, step)
            steps_executed += 1
            if src and src not in sources:
                sources.append(src)
            if outcome.startswith("[error") or outcome.startswith("[timeout"):
                errors.append(outcome)
            else:
                # Only collect meaningful extracts
                if step.action == "extract" and outcome:
                    raw_extracts.append(outcome)

        notes = "\n\n".join(raw_extracts) if raw_extracts else "No extractions captured."
        summary = await planner.summarize(notes)

        await context.close()
        await browser.close()

        return AgentResult(
            agent_name=agent_name,
            nav_prompt=nav_prompt,
            steps_executed=steps_executed,
            summary=summary,
            sources=sources,
            errors=errors,
            raw_extracts=raw_extracts,
        )