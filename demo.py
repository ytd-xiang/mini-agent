"""
ReAct Agent Demo — Qwen2.5-1.5B-Instruct

Task suite:
  1. Simple tool use (datetime)
  2. Calculator
  3. Knowledge retrieval
  4. Multi-step reasoning
  5. Code execution

With structured result reporting.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from agent import ReActAgent


def demo_task(agent, task, label):
    print(f"\n{'#'*55}")
    print(f"# {label}")
    print(f"{'#'*55}")
    result = agent.run(task)

    # Summary card
    status = "OK" if result.success else "FAIL"
    print(f"\n  {'─'*50}")
    print(f"  {label} | Status: {status}")
    print(f"  Tool calls: {result.total_tool_calls} | Steps: {len(result.steps)}")
    if result.answer:
        print(f"  Answer: {result.answer[:200]}")
    if result.error:
        print(f"  Error: {result.error}")

    # Tool trace
    if result.total_tool_calls > 0:
        print(f"\n  Tool trace:")
        for s in result.steps:
            if s.action and s.action != "Final Answer":
                obs = (s.observation or "")[:80].replace("\n", " | ")
                print(f"    [{s.step_num}] {s.action}({s.action_input[:50]})")
                print(f"         -> {obs}")


def main():
    print("=" * 55)
    print("  ReAct Agent — Full ReAct Loop")
    print("  Model: Qwen2.5-1.5B-Instruct")
    print("  Tools: calculator, datetime, search, code_executor, word_count")
    print("=" * 55)

    agent = ReActAgent(max_steps=5, verbose=True)

    demo_task(agent, "今天是几月几号？星期几？", "Task 1: Datetime")
    demo_task(agent, "计算 (156 + 89) * 4 - 200 等于多少", "Task 2: Calculator")
    demo_task(agent, "LoRA是什么技术？简要说明。", "Task 3: Knowledge Search")
    demo_task(agent, "搜索Transformer的信息，然后告诉我它是哪一年提出的。", "Task 4: Multi-step")
    demo_task(agent, "写Python代码计算斐波那契数列前10项，用code_executor执行。", "Task 5: Code Exec")

    print(f"\n{'='*55}")
    print(f"  All tasks complete!")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
