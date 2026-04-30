"""
CLI entry point for Agent Core Architecture.
Provides: run, dry-run, replay-trace commands.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.execution import AgentOrchestrator
from agent.tracing import TraceReplayer


def get_default_paths() -> tuple[Path, Path]:
    workspace = Path.cwd()
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_path = data_dir / f"memory_{timestamp}.json"
    trace_path = data_dir / f"trace_{timestamp}.jsonl"

    return memory_path, trace_path


def cmd_run(args: argparse.Namespace) -> None:
    memory_path, trace_path = get_default_paths()

    if args.memory_path:
        memory_path = Path(args.memory_path)
    if args.trace_path:
        trace_path = Path(args.trace_path)

    orchestrator = AgentOrchestrator(
        memory_persistent_path=str(memory_path),
        trace_output_path=str(trace_path),
        max_steps=args.max_steps,
        tool_timeout=args.timeout,
        planner_max_retries=args.max_retries,
        planner_abort_threshold=args.abort_threshold,
    )

    print("=" * 80)
    print("AGENT EXECUTION (RUN MODE)")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Memory file: {memory_path}")
    print(f"Trace file: {trace_path}")
    print("-" * 80)

    result = orchestrator.run(
        user_input_text=args.input,
        dry_run=False,
    )

    print("-" * 80)
    print("EXECUTION RESULT")
    print("-" * 80)
    print(f"Success: {result.success}")
    print(f"Steps executed: {result.steps_executed}")
    print(f"Final answer: {result.final_answer}")
    if result.failure_reason:
        print(f"Failure reason: {result.failure_reason}")
    print(f"Trace file: {result.trace_file_path}")
    print("=" * 80)


def cmd_dry_run(args: argparse.Namespace) -> None:
    _, trace_path = get_default_paths()

    if args.trace_path:
        trace_path = Path(args.trace_path)

    orchestrator = AgentOrchestrator(
        trace_output_path=str(trace_path),
        max_steps=args.max_steps,
    )

    print("=" * 80)
    print("AGENT EXECUTION (DRY-RUN MODE)")
    print("=" * 80)
    print(f"Input: {args.input}")
    print("Note: Tools will NOT be actually called in dry-run mode.")
    print("-" * 80)

    result = orchestrator.run(
        user_input_text=args.input,
        dry_run=True,
    )

    print("-" * 80)
    print("DRY-RUN RESULT")
    print("-" * 80)
    print(f"Final answer: {result.final_answer}")
    print("=" * 80)


def cmd_replay_trace(args: argparse.Namespace) -> None:
    trace_file = Path(args.trace_file)

    if not trace_file.exists():
        print(f"Error: Trace file not found: {trace_file}")
        sys.exit(1)

    print("=" * 80)
    print("TRACE REPLAY")
    print("=" * 80)
    print(f"Trace file: {trace_file}")
    print("-" * 80)

    events = TraceReplayer.replay(str(trace_file))
    summary = TraceReplayer.format_summary(events)

    print(summary)
    print("=" * 80)


def cmd_demo(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("AGENT CORE ARCHITECTURE - DEMO")
    print("=" * 80)

    memory_path, trace_path = get_default_paths()
    orchestrator = AgentOrchestrator(
        memory_persistent_path=str(memory_path),
        trace_output_path=str(trace_path),
        max_steps=10,
        tool_timeout=5.0,
        planner_max_retries=2,
        planner_abort_threshold=2,
    )

    print("\n[DEMO 1] Successful Tool Call - Calculator")
    print("-" * 80)
    print("Input: 'Calculate 15 plus 7'")
    result1 = orchestrator.run("Calculate 15 plus 7")
    print(f"\nResult: {result1.final_answer}")
    print(f"Success: {result1.success}")

    print("\n" + "=" * 80)
    print("[DEMO 2] Memory Write and Read")
    print("-" * 80)
    print("First, save something to memory...")
    print("Input: 'Remember that my favorite color is blue'")
    result2 = orchestrator.run("Remember that my favorite color is blue")
    print(f"\nResult: {result2.final_answer}")
    print(f"Success: {result2.success}")

    print("\nNow, try to recall it...")
    print("Input: 'Recall my favorite color'")
    result3 = orchestrator.run("Recall my favorite color")
    print(f"\nResult: {result3.final_answer}")
    print(f"Success: {result3.success}")

    print("\n" + "=" * 80)
    print("[DEMO 3] Tool Failure Handling - Unstable API")
    print("-" * 80)
    print("Input: 'Search for information about AI (this may fail)'")
    print("(Note: The unstable API has a 30% failure rate and 20% timeout rate)")

    result4 = orchestrator.run("Search for information about AI")
    print(f"\nResult: {result4.final_answer}")
    print(f"Success: {result4.success}")
    if result4.failure_reason:
        print(f"Failure reason: {result4.failure_reason}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"Trace file saved to: {trace_path}")
    print(f"Memory file saved to: {memory_path}")
    print("\nTo replay the trace, run:")
    print(f"  python -m cli.main replay-trace --trace-file {trace_path}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Core Architecture CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run the agent with input")
    run_parser.add_argument("--input", "-i", required=True, help="User input text")
    run_parser.add_argument("--memory-path", type=str, help="Path to memory file")
    run_parser.add_argument("--trace-path", type=str, help="Path to trace output file")
    run_parser.add_argument("--max-steps", type=int, default=10, help="Maximum execution steps")
    run_parser.add_argument("--timeout", type=float, default=30.0, help="Tool timeout in seconds")
    run_parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per action")
    run_parser.add_argument("--abort-threshold", type=int, default=3, help="Consecutive failures before abort")

    dry_run_parser = subparsers.add_parser("dry-run", help="Generate plan without executing tools")
    dry_run_parser.add_argument("--input", "-i", required=True, help="User input text")
    dry_run_parser.add_argument("--trace-path", type=str, help="Path to trace output file")
    dry_run_parser.add_argument("--max-steps", type=int, default=10, help="Maximum execution steps")

    replay_parser = subparsers.add_parser("replay-trace", help="Replay and visualize a trace file")
    replay_parser.add_argument("--trace-file", "-f", required=True, help="Path to JSONL trace file")

    demo_parser = subparsers.add_parser("demo", help="Run a demonstration of all features")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "dry-run":
        cmd_dry_run(args)
    elif args.command == "replay-trace":
        cmd_replay_trace(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
