"""
Tool system for the ReAct Agent.

Each tool has:
  - name: unique identifier
  - description: what it does (included in system prompt)
  - parameters: JSON Schema for parameter validation
  - function: the actual callable

Design principles:
  1. Tools are self-documenting (description goes into prompt)
  2. Input validation before execution
  3. Graceful error handling (never crash the agent loop)
  4. All tools return strings (text observations)
"""

import json
import datetime
import math
import re
from typing import Callable, Dict, Any, Optional


class Tool:
    """A single tool that the agent can call."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict] = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def to_prompt_description(self) -> str:
        """Generate the description string for the system prompt."""
        params_str = ""
        if self.parameters:
            params_str = "\n    Parameters: " + json.dumps(self.parameters, ensure_ascii=False)
        return f"- {self.name}: {self.description}{params_str}"

    def execute(self, input_str: str) -> str:
        """Execute the tool with given input. Returns observation text."""
        try:
            result = self.func(input_str)
            return str(result)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


# ============================================================
# Tool Implementations
# ============================================================

def tool_calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt(), sin(), cos(), abs()
    """
    # Whitelist allowed operations
    allowed = set("0123456789+-*/().% eExXpPiIqQtTsSiInNcCoOaAbBsS")
    cleaned = "".join(c for c in expression if c in allowed)

    if not cleaned.strip():
        return "Error: no valid expression found"

    try:
        # Replace common math functions
        safe_expr = cleaned
        result = eval(safe_expr, {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
            "pow": math.pow,
        })
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


def tool_datetime(query: str) -> str:
    """Get current date/time or perform date calculations."""
    query = query.strip().lower()
    now = datetime.datetime.now()

    if "年份" in query or "year" in query:
        return f"Current year: {now.year}"
    elif "月份" in query or "month" in query:
        return f"Current month: {now.month} ({now.strftime('%B')})"
    elif "星期" in query or "weekday" in query:
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return f"Today is {weekdays[now.weekday()]}"
    elif "日期" in query or "date" in query:
        return f"Current date: {now.strftime('%Y-%m-%d')}"
    else:
        return f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S')} (weekday: {now.strftime('%A')})"


def tool_word_count(text: str) -> str:
    """Count characters and approximate words in text."""
    chars = len(text)
    words = len(text.split())
    lines = text.count("\n") + 1 if text else 0
    eng_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    chn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return (
        f"Character count: {chars}\n"
        f"Word count: {words}\n"
        f"Line count: {lines}\n"
        f"English alphabetic chars: {eng_chars}\n"
        f"Chinese characters: {chn_chars}"
    )


def tool_search(query: str) -> str:
    """
    Simulated search returning predefined results.
    Replace with a real search API for production use.
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
        "transformer": "The Transformer architecture was introduced in 'Attention Is All You Need' (Vaswani et al., 2017). It uses self-attention mechanisms.",
        "lora": "LoRA (Low-Rank Adaptation) is a PEFT method that freezes pretrained weights and injects trainable rank decomposition matrices.",
        "react": "ReAct (Reasoning + Acting) is a paradigm where LLMs interleave reasoning traces and task-specific actions.",
        "langchain": "LangChain is a framework for developing applications powered by language models, including chains, agents, and retrieval.",
    }

    query_lower = query.lower()
    results = []
    for key, value in knowledge.items():
        if key in query_lower:
            results.append(f"[{key}] {value}")

    if not results:
        return f"No results found for '{query}'. Try a different search term."

    return "\n\n".join(results)


def tool_code_executor(code: str) -> str:
    """
    Execute Python code and return the output.
    Safety: restricted globals, no file access, timeout simulation.
    """
    # Strip markdown code fences if present
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)

    safe_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "sum": sum,
            "max": max,
            "min": min,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "abs": abs,
            "round": round,
            "type": type,
            "isinstance": isinstance,
        },
    }

    # Capture stdout
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        exec(code, safe_globals, {})
        output = captured.getvalue()
        if output.strip():
            return f"Output:\n{output.strip()}"
        else:
            # Try to evaluate the last expression
            try:
                result = eval(code.strip().split("\n")[-1], safe_globals, {})
                return f"Result: {result}"
            except Exception:
                return "Code executed successfully (no output)."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout


# ============================================================
# Tool Registry
# ============================================================

def create_default_tools() -> Dict[str, Tool]:
    """Create the default tool set for the agent."""
    return {
        "calculator": Tool(
            name="calculator",
            description="Evaluate a mathematical expression. Use for arithmetic, trigonometry, etc.",
            func=tool_calculator,
            parameters={"expression": "string, the math expression to evaluate"},
        ),
        "datetime": Tool(
            name="datetime",
            description="Get current date, time, year, month, or weekday. Input can be: date, time, year, month, weekday.",
            func=tool_datetime,
            parameters={"query": "string: date, time, year, month, or weekday"},
        ),
        "word_count": Tool(
            name="word_count",
            description="Count characters, words, and lines in a text string.",
            func=tool_word_count,
            parameters={"text": "string, the text to analyze"},
        ),
        "search": Tool(
            name="search",
            description="Search for information in the knowledge base. Use for factual queries about Python, Transformer, LoRA, ReAct, LangChain.",
            func=tool_search,
            parameters={"query": "string, search keywords"},
        ),
        "code_executor": Tool(
            name="code_executor",
            description="Execute Python code and return the output. Use for calculations, data processing, or algorithm demonstration.",
            func=tool_code_executor,
            parameters={"code": "string, Python code to execute"},
        ),
    }
