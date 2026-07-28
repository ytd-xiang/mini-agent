# Mini Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> ReAct Agent implementation powered by Qwen2.5-1.5B-Instruct. Built without LangChain or any agent framework.

## Architecture

```
User Task
    │
    ▼
┌─────────────────────────────────────┐
│          ReAct Agent                 │
│                                     │
│  ┌───────────────────────────────┐  │
│  │   Qwen2.5-1.5B-Instruct       │  │
│  │                               │  │
│  │   Thought: "需要计算表达式"     │  │
│  │   Action: calculator          │  │
│  │   Action Input: (156+89)*4   │  │
│  └───────────┬───────────────────┘  │
│              │                      │
│              ▼                      │
│  Observation: "Result: 980.0"       │
│              │                      │
│              ▼                      │
│  Thought: "可以回答"                 │
│  Final Answer: 980                  │
│              │                      │
│     ┌────────┼────────┬────────┐    │
│     ▼        ▼        ▼        ▼    │
│  calculator datetime search code   │
└─────────────────────────────────────┘
```

## Quick Start

```bash
pip install torch transformers
python demo.py
```

Requires Qwen2.5-1.5B-Instruct cached locally (~3GB).

## Tools

| Tool | Description |
|:-----|:-----------|
| `calculator` | Evaluate mathematical expressions |
| `datetime` | Get current date, time, year, month, weekday |
| `search` | Query built-in knowledge base |
| `code_executor` | Execute Python code in sandbox |
| `word_count` | Count characters, words, lines in text |

## Usage

```python
from agent import ReActAgent

agent = ReActAgent(max_steps=5, verbose=True)
result = agent.run("What is the date today?")

print(result.answer)        # "2026-07-28, Monday"
print(result.steps)         # Full Thought → Action → Observation chain
print(result.total_tool_calls)  # Number of tool invocations
```

### Register custom tools

```python
from tools import Tool

def get_weather(city: str) -> str:
    return f"{city}: sunny, 25°C"

agent.tools["weather"] = Tool(
    name="weather",
    description="Query weather for a city",
    func=get_weather,
)
```

## Key Features

- Full ReAct loop: Thought → Action → Observation → Final Answer
- Structured output parsing with fallback strategies
- Loop detection: auto-break on repeated identical tool calls
- Max step guard: prevents infinite execution
- Extensible tool system: register new tools via simple interface
- Three LLM backends: local model, API, offline mode

## Project Structure

```
mini-agent/
├── agent.py      # ReAct loop and LLM integration
├── tools.py      # Tool base class and 5 built-in tools
├── demo.py       # 5-task demonstration
└── README.md
```

## License

MIT
