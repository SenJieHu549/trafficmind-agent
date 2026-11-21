import { test } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { loadTs } from './loadTs.mjs';
const { collabApi } = loadTs(new URL('../src/api/collaborationApi.ts', import.meta.url));
const { reduceCollaborationEvent } = loadTs(new URL('../src/utils/collaborationEventReducer.ts', import.meta.url));
const { default: Card } = loadTs(new URL('../src/components/collaboration/HumanReviewCard.tsx', import.meta.url));

const review = { reviewId: 'review', runId: 'run', version: 1, status: 'pending', decision: {},
  conflicts: [{ id: 'c1', description: '信号与过街冲突', severity: 'high', agents: ['SignalAgent'],
    proposals: [{ agentName: 'SignalAgent', suggestion: '保持行人过街相位', evidenceRefs: ['E1'], resultVersion: 'v1' }] }] };

test('waiting SSE and transport done preserve a resumable, unfinished run', () => {
  let state = { status: 'arbitrating', tasks: [], agentResults: {}, completedAt: '' };
  state = reduceCollaborationEvent(state, { eventType: 'run_awaiting_review', reviewId: 'review' });
  state = reduceCollaborationEvent(state, { eventType: 'done', status: 'awaiting_review', timestamp: 'now' });
  assert.equal(state.status, 'awaiting_review');
  assert.equal(state.requiresHumanReview, true);
  assert.equal(state.completedAt, '');
});

test('decision API binds run, review identity and expected version', async () => {
  const original = globalThis.fetch;
  const body = { expectedVersion: 1, action: 'reject', reviewer: 'tester', comment: 'Missing evidence', resolutions: [] };
  globalThis.fetch = async (url, options) => {
    assert.match(url, /runs\/run%2F1\/reviews\/review%3F2$/);
    assert.deepEqual(JSON.parse(options.body), body);
    assert.equal(options.method, 'POST');
    return Response.json({ review: { ...review, status: 'rejected', version: 2 }, duplicate: false });
  };
  try { assert.equal((await collabApi.decideReview('run/1', 'review?2', body)).review.status, 'rejected'); }
  finally { globalThis.fetch = original; }
});

test('stale review error is displayed as a concrete server message', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ detail: { code: 'review_version_conflict', message: '版本已变化，请刷新' } }, { status: 409 });
  try { await assert.rejects(collabApi.decideReview('run', 'review', {}), /版本已变化/); }
  finally { globalThis.fetch = original; }
});

test('review card presents competing source evidence and requires an explicit decision', () => {
  const html = renderToStaticMarkup(React.createElement(Card, { review }));
  assert.match(html, /保持行人过街相位/);
  assert.match(html, /E1/);
  assert.match(html, /外部处置动作仍需另行审批/);
  assert.match(html, /disabled="">确认裁决并继续融合/);
  assert.match(html, /裁决处理人/);
});

test('decided card does not offer another confirmation form', () => {
  const html = renderToStaticMarkup(React.createElement(Card, { review: { ...review, status: 'accepted', version: 2,
    decision: { reviewer: 'tester', comment: 'Confirmed' } } }));
  assert.match(html, /已确认/);
  assert.match(html, /tester/);
  assert.doesNotMatch(html, /确认裁决并继续融合/);
});
