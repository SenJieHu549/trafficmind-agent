"""Process-local producer survives a disconnected SSE subscriber.

This is not a distributed/restartable worker. Durable runs/tasks remain the
queryable source of truth; the queue is bounded transport buffering only.
"""
import asyncio
import json
from backend.agent.streaming import sse_event

_producers = set()


async def detached_stream(source, *, timeout, heartbeat=10):
    queue = asyncio.Queue(maxsize=256)
    attached = True
    identity = {}

    def push(value):
        if attached:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(value)

    async def produce():
        try:
            async with asyncio.timeout(timeout):
                async for frame in source:
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                for key in ("runId", "sessionId"):
                                    if data.get(key):
                                        identity[key] = data[key]
                            except (TypeError, ValueError):
                                pass
                    push(frame)
        except (Exception, asyncio.CancelledError) as exc:
            code = "collaboration_timeout" if isinstance(exc, TimeoutError) else (
                "server_task_cancelled" if isinstance(exc, asyncio.CancelledError) else "collaboration_" + type(exc).__name__)
            if identity.get("runId"):
                from backend.agent.collaboration.db_repository import SQLiteCollaborationRepository
                repo = SQLiteCollaborationRepository()
                run = repo.get_run(identity["runId"])
                if run and run["status"] == "awaiting_review":
                    # A transport/post-processing exception cannot undo the
                    # durable pause or turn it into an analysis failure.
                    push(sse_event("run_awaiting_review", {**identity, "status": "awaiting_review"}))
                    push(sse_event("done", {**identity, "status": "awaiting_review"}))
                    return
                if run and run["status"] not in ("completed", "partial_success", "failed", "interrupted", "awaiting_review"):
                    run.update(status="failed", final_decision={"failureReason": code})
                    repo.update_run(run)
                repo.save_event(identity["runId"], {"event_id": "background_failure",
                    "event_type": "run_failed", "code": code}, 0)
            push(sse_event("run_failed", {**identity, "reason": code}))
            push(sse_event("done", {**identity, "status": "failed"}))
        finally:
            try:
                await source.aclose()
            finally:
                push(None)

    producer = asyncio.create_task(produce(), name="collaboration-producer")
    _producers.add(producer)
    producer.add_done_callback(_producers.discard)
    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if frame is None:
                break
            yield frame
    finally:
        # Do not cancel producer on an HTTP disconnect. It has its own deadline.
        attached = False
