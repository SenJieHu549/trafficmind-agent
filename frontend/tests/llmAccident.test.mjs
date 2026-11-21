import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadTs } from './loadTs.mjs';
const { collabApi } = loadTs(new URL('../src/api/collaborationApi.ts', import.meta.url));
const { reduceCollaborationEvent } = loadTs(new URL('../src/utils/collaborationEventReducer.ts', import.meta.url));

async function stream(chunks, events) {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(new ReadableStream({ start(c) {
    for (const s of chunks) c.enqueue(new TextEncoder().encode(s)); c.close();
  }}));
  try { await collabApi.streamCollaboration({}, { onEvent: e => events.push(e) }); }
  finally { globalThis.fetch = original; }
}

test('collaboration SSE preserves tool result identity across network chunks', async () => {
  const events = [];
  await stream(['event: run_created\n', 'data: {"runId":"r"}\n\n', ': keepalive\n\n',
    'event: agent_result\n', 'data: {"agentName":"AccidentAgent","executionMode":"llm_tools","runtimeTrace":{"toolRounds":2},"result":{"findings":[]}}\n\n',
    'event: run_completed\n', 'data: {"runId":"r"}\n\n'], events);
  assert.deepEqual(events.map(e => e.eventType), ['run_created', 'agent_result', 'run_completed']);
  assert.equal(events[1].executionMode, 'llm_tools');
  assert.equal(events[1].runtimeTrace.toolRounds, 2);
});

test('disconnect retains durable run identity and does not invent a terminal state', async () => {
  const events = [];
  await stream(['event: run_created\ndata: {"runId":"r"}\n\n'], events);
  assert.equal(events.at(-1).eventType, 'transport_disconnected');
  assert.equal(events.at(-1).runId, 'r');
  const state = { runId: 'r', status: 'running' };
  assert.equal(reduceCollaborationEvent(state, events.at(-1)).status, 'running');
});

test('LLM fallback and partial status remain explicit in frontend state', () => {
  const state = { tasks: [], agentResults: {} };
  const next = reduceCollaborationEvent(state, { eventType: 'agent_result', agentName: 'AccidentAgent',
    status: 'partial', executionMode: 'rule_fallback', degraded: true, fallbackReason: 'llm_timeout',
    runtimeTrace: { toolRounds: 0 }, result: { findings: ['known facts'], confidence: .5, recommendation: 'review' } });
  assert.equal(next.agentResults.AccidentAgent.executionMode, 'rule_fallback');
  assert.equal(next.agentResults.AccidentAgent.status, 'partial');
  assert.equal(next.agentResults.AccidentAgent.fallbackReason, 'llm_timeout');
});
