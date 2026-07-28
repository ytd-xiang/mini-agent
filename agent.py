"""
ReAct Agent — Qwen2.5-1.5B-Instruct.

ReAct loop: Thought -> Action -> Observation -> repeat -> Final Answer.

Architecture:
  User Task -> LLM (Thought + Action) -> Tool Execution -> Observation
       ^                                                    |
       └────────────────────────────────────────────────────┘
                          (loop until Final Answer)
"""

import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import Tool, create_default_tools

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ============================================================
# Data Structures
# ============================================================

@dataclass
class AgentStep:
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""

@dataclass
class AgentResult:
    success: bool
    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    total_tool_calls: int = 0
    error: Optional[str] = None


# ============================================================
# System Prompt — ReAct format for Qwen2.5
# ============================================================

def build_system_prompt(tools: Dict[str, Tool]) -> str:
    """Build concise system prompt optimized for Qwen2.5-1.5B."""

    tool_lines = []
    for t in tools.values():
        tool_lines.append(f"{t.name}: {t.description}")


    return f"""你能使用以下工具：

{chr(10).join(tool_lines)} 

使用工具时按以下格式：
Thought: 分析当前情况
Action: 工具名
Action Input: 输入内容

获得足够信息后：
Thought: 可以回答了
Final Answer: 最终答案

规则：每步一个工具，用工具获取事实不要猜测。"""


# ============================================================
# Output Parser
# ============================================================

def parse_react_output(text: str) -> Tuple[bool, str, str]:
    """
    Parse ReAct format from model output.
    Uses LAST matching pattern (model might echo template, then do real work).
    """
    # Find ALL Final Answer/Action matches, use the LAST one
    final_matches = list(re.finditer(r"Final Answer:\s*([\s\S]+?)$", text, re.MULTILINE))
    action_matches = list(re.finditer(
        r"^Action:\s*(\S+)\s*$\n^Action Input:\s*(.+?)$", text, re.MULTILINE
    ))

    # If both exist, compare positions — use the one that appears last
    last_final_pos = final_matches[-1].start() if final_matches else -1
    last_action_pos = action_matches[-1].start() if action_matches else -1

    if last_final_pos > last_action_pos:
        return True, final_matches[-1].group(1).strip(), ""
    elif last_action_pos > last_final_pos:
        return False, action_matches[-1].group(1).strip(), action_matches[-1].group(2).strip()

    # Single match logic
    if final_matches:
        return True, final_matches[-1].group(1).strip(), ""
    if action_matches:
        return False, action_matches[-1].group(1).strip(), action_matches[-1].group(2).strip()

    # Fallback: single-line Action
    action_simple = list(re.finditer(r"^Action:\s*(\S+)", text, re.MULTILINE))
    if action_simple:
        return False, action_simple[-1].group(1).strip(), ""

    # Text too short -> retry
    cleaned = text.strip()
    if len(cleaned) > 30:
        return True, cleaned, ""

    return True, "格式错误，请重试。", ""


# ============================================================
# ReAct Agent
# ============================================================

class ReActAgent:
    """
    ReAct Agent powered by Qwen2.5-1.5B-Instruct.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        tools: Optional[Dict[str, Tool]] = None,
        max_steps: int = 6,
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.tools = tools or create_default_tools()
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Agent] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[Agent] Ready. {len(self.tools)} tools: {list(self.tools.keys())}")

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate model response from chat messages."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "".join(prompt_parts)

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                top_p=0.9 if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|im_start|>assistant" in raw:
            parts = raw.split("<|im_start|>assistant")
            return parts[-1].strip()

        return raw.strip()

    def run(self, task: str) -> AgentResult:
        """Execute the ReAct loop."""

        system_prompt = build_system_prompt(self.tools)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        steps = []
        tool_calls = 0

        if self.verbose:
            print(f"\n{'='*55}")
            print(f"  Task: {task}")
            print(f"{'='*55}")

        for i in range(self.max_steps):
            response = self._generate(messages)
            is_final, action_or_answer, action_input = parse_react_output(response)

            # Extract thought
            thought_m = re.search(r"Thought:\s*([\s\S]+?)(?=\n(?:Action|Final)|$)", response)
            thought = thought_m.group(1).strip() if thought_m else ""

            step = AgentStep(step_num=i + 1, thought=thought)

            if self.verbose:
                print(f"\n  --- Step {i + 1} ---")
                if thought:
                    print(f"  Thought: {thought[:200]}")
                if is_final:
                    print(f"  >> Final Answer: {action_or_answer[:200]}")
                else:
                    print(f"  Action: {action_or_answer}")
                    print(f"  Input:  {action_input[:100]}")

            if is_final:
                step.action = "Final Answer"
                step.action_input = action_or_answer
                steps.append(step)
                return AgentResult(
                    success=True,
                    answer=action_or_answer,
                    steps=steps,
                    total_tool_calls=tool_calls,
                )

            # Execute tool
            action_name = action_or_answer
            step.action = action_name
            step.action_input = action_input

            if action_name in self.tools:
                tool_calls += 1
                observation = self.tools[action_name].execute(action_input)
                step.observation = observation

                if self.verbose:
                    print(f"  Observation: {observation[:200]}")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"工具返回: {observation}"})

                # Loop detection
                if i >= 3:
                    recent = [s.action for s in steps[-3:] if s.action != "Final Answer"]
                    if len(recent) >= 3 and len(set(recent)) == 1:
                        messages.append({
                            "role": "user",
                            "content": "你已经连续多次使用相同工具。请直接给出Final Answer。"
                        })
            else:
                error_msg = f"工具 '{action_name}' 不存在。可用: {list(self.tools.keys())}"
                step.observation = error_msg
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": error_msg})

                if self.verbose:
                    print(f"  Error: {error_msg}")

            steps.append(step)

        # Max steps reached
        messages.append({"role": "user", "content": "已达最大步数。直接给出Final Answer。"})
        final_response = self._generate(messages)
        is_final, answer, _ = parse_react_output(final_response)

        return AgentResult(
            success=is_final,
            answer=answer if is_final else f"未能在{self.max_steps}步内完成任务。",
            steps=steps,
            total_tool_calls=tool_calls,
        )


# ============================================================
# Quick start
# ============================================================

def run_agent(task: str) -> AgentResult:
    agent = ReActAgent()
    return agent.run(task)
