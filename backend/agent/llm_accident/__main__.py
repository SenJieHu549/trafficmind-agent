"""python -m backend.agent.llm_accident --event-id ... --output-dir D:/..."""
import argparse
import asyncio
import json

from backend.agent.llm_accident.runtime import AccidentAgentRuntime
from backend.agent.llm_accident.contracts import RuntimeLimits


def main():
    from backend.config import ACCIDENT_OUTPUT_REPAIR_ATTEMPTS
    parser = argparse.ArgumentParser(description="Standalone read-only accident Agent; does not dispatch a Workflow")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--mode", choices=("rule", "llm_context", "llm_tools"), default="llm_tools")
    parser.add_argument("--question", default="请分析当前事故，按需查证并给出有依据的处置建议。")
    parser.add_argument("--grounding-run", help="Optional completed collaboration run supplying persisted initial knowledge")
    parser.add_argument("--output-dir", required=True, help="Append-only run/evidence trace directory")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--output-repairs",type=int,choices=(0,1),default=ACCIDENT_OUTPUT_REPAIR_ATTEMPTS,
                        help="Optional one-shot format repair within the original call/time budgets")
    parser.add_argument("--tool-timeout", type=float, default=240, help="Per-query wait deadline in seconds")
    parser.add_argument("--overall-timeout", type=float, default=360, help="Whole analysis deadline in seconds")
    args = parser.parse_args()
    grounding = None
    if args.grounding_run:
        from backend.agent.collaboration.db_repository import SQLiteCollaborationRepository
        run = SQLiteCollaborationRepository().get_run(args.grounding_run)
        if not run or run.get("status") != "completed":
            parser.error("grounding source must be a completed persisted run")
        grounding = json.loads(run["grounding_context"]) if isinstance(run["grounding_context"], str) else run["grounding_context"]
        if (grounding.get("currentEvent") or {}).get("eventId") != args.event_id:
            parser.error("grounding source belongs to a different event")
    limits = RuntimeLimits(tool_timeout=args.tool_timeout, overall_timeout=args.overall_timeout,
                           max_output_repairs=args.output_repairs)
    report = asyncio.run(AccidentAgentRuntime(limits=limits).run(args.event_id, question=args.question, grounding=grounding,
                        mode=args.mode, fallback=not args.no_fallback, output_dir=args.output_dir))
    trace = report["trace"]
    print(json.dumps({"runId": trace["runId"], "status": trace["status"],
        "generationMode": trace["generationMode"], "failureReason": trace["failureReason"],
        "llmCalls": len(trace["llmCalls"]), "toolRounds": trace["toolRounds"],
        "toolAttempts": trace["toolAttempts"], "durationMs": trace["durationMs"],
        "outputRepairAttempts":trace["outputRepairAttempts"],"outputRepairValidated":trace["outputRepairValidated"],
        "usage": trace["usage"], "evidenceCount": len((report["evidenceSnapshot"] or {}).get("evidence", []))}, ensure_ascii=False))
    if not report["agentResult"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
