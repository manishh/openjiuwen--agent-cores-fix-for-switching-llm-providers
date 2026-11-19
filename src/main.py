"""Main entry point: runs all three provider-swap demonstration scenarios.

Usage:
    python -m src.main

Each scenario is self-contained; you can also run each file individually:
    python src/baseline_agent.py
    python src/provider_swap.py
    python src/multi_model.py
"""
import asyncio
import os
import sys


def _require_env(names: list[str]) -> None:
    """Abort early with a clear message if any required env-var is missing."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


async def main() -> None:
    print("=" * 60)
    print("SCENARIO 1 — Baseline WorkflowAgent (OpenAI)")
    print("=" * 60)
    _require_env(["OPENAI_API_KEY"])
    from src.baseline_agent import run_baseline
    await run_baseline()

    print()
    print("=" * 60)
    print("SCENARIO 2 — Provider Swap (DeepSeek, identical workflow)")
    print("=" * 60)
    _require_env(["DEEPSEEK_API_KEY"])
    from src.provider_swap import run_swap
    await run_swap()

    print()
    print("=" * 60)
    print("SCENARIO 3 — Multi-Model Config (select provider at runtime)")
    print("=" * 60)
    _require_env(["OPENAI_API_KEY", "DEEPSEEK_API_KEY"])
    from src.multi_model import run_multi_model
    await run_multi_model()


if __name__ == "__main__":
    asyncio.run(main())
