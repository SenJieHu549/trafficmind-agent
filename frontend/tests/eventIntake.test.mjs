import { test } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { loadTs } from './loadTs.mjs';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const { eventIntakeApi } = loadTs(new URL('../src/api/eventIntakeApi.ts', import.meta.url));
const { default: Panel, IntakeFactsFields, IntakeConfirmationSummary } = loadTs(new URL('../src/components/events/EventIntakePanel.tsx', import.meta.url));
const stateSource = readFileSync(new URL('../src/components/simulation/eventWorkbenchState.ts', import.meta.url), 'utf8');
const stateJs = ts.transpileModule(stateSource, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText;
const { eventSourceLabel } = await import(`data:text/javascript;base64,${Buffer.from(stateJs).toString('base64')}`);

test('preview only sends a draft request and never sends an analysis request', async () => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push(url);
    assert.deepEqual(JSON.parse(options.body), { description: '学校附近两车相撞，伤亡未知' });
    return Response.json({ facts: {} });
  };
  try { await eventIntakeApi.preview('学校附近两车相撞，伤亡未知'); }
  finally { globalThis.fetch = original; }
  assert.deepEqual(calls, ['/api/events/intake/preview']);
});

test('confirmation preserves null versus zero, scope and stable retry identity', async () => {
  const original = globalThis.fetch;
  const body = { requestId: 'stable-request', confirmed: true, regionId: 'r', roadId: 'road',
    description: 'collision', dataReality: 'synthetic', facts: { injuredCount: null, vehicleCount: 2, avgSpeed: 0 } };
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/events/intake/confirm');
    assert.deepEqual(JSON.parse(options.body), body);
    return Response.json({ eventId: 'E1', analysisStarted: false,
      next: { analysisUrl: '/?view=multi&eventId=E1', eventUrl: '/?view=simulation&eventId=E1' } });
  };
  try {
    const result = await eventIntakeApi.confirm(body);
    assert.equal(result.analysisStarted, false);
    assert.match(result.next.analysisUrl, /eventId=E1/);
  } finally { globalThis.fetch = original; }
});

test('server validation errors surface to the confirmation form', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ detail: { message: '请选择同一区域内有效的规范道路' } }, { status: 422 });
  try { await assert.rejects(eventIntakeApi.confirm({}), /有效的规范道路/); }
  finally { globalThis.fetch = original; }
});

test('registration clearly separates preview, confirmation and paid analysis', () => {
  const html = renderToStaticMarkup(React.createElement(Panel));
  assert.match(html, /事件登记/);
  assert.match(html, /解析事件/);
  assert.match(html, /确认登记后进入协同研判/);
  assert.match(html, /disabled=""[^>]*>解析事件/);
});

test('editable facts display unknown counts empty and explicit zero as zero', () => {
  const html = renderToStaticMarkup(React.createElement(IntakeFactsFields, { facts: {
    eventType: 'accident', vehicleCount: 2, injuredCount: null, fatalityCount: 0,
    avgSpeed: null, queueLength: null, duration: null, nearbySchool: true,
    nearbyHospital: false, weather: 'unknown', timePeriod: 'unknown',
  }, onChange: () => {} }));
  assert.match(html, /受伤人数[^<]*<input[^>]*value=""/);
  assert.match(html, /死亡人数[^<]*<input[^>]*value="0"/);
  assert.match(html, /平均车速[^<]*<input[^>]*value=""/);
});

test('manual synthetic events and unverified reports have distinct provenance labels', () => {
  assert.equal(eventSourceLabel({ source: 'manual_confirmed_report', datasetReality: 'synthetic' }), '人工登记 · 演示数据');
  assert.equal(eventSourceLabel({ source: 'manual_confirmed_report' }), '人工上报 · 待核实');
});

test('actual numeric input handlers preserve explicit zero and return cleared fields to null', async () => {
  let facts = { eventType: 'accident', vehicleCount: 2, injuredCount: null, fatalityCount: null,
    avgSpeed: null, queueLength: null, duration: 300, isMainRoad: true,
    nearbySchool: true, nearbyHospital: false, weather: 'unknown', timePeriod: 'unknown' };
  const edit = (key, value) => {
    const tree = IntakeFactsFields({ facts, onChange: next => { facts = next; } });
    const label = tree.props.children.flat().find(child => child?.key === key);
    const input = label.props.children.find(child => child?.type === 'input');
    input.props.onChange({ target: { value } });
  };
  edit('injuredCount', '0');
  assert.equal(facts.injuredCount, 0);
  edit('injuredCount', '');
  assert.equal(facts.injuredCount, null);
  edit('duration', '5');
  const original = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    const sent = JSON.parse(options.body);
    assert.equal(sent.facts.injuredCount, null);
    assert.equal(sent.facts.fatalityCount, null);
    assert.equal(sent.facts.duration, 5);
    assert.equal(sent.facts.isMainRoad, true);
    return Response.json({ eventId: 'E1', analysisStarted: false });
  };
  try { await eventIntakeApi.confirm({ facts }); } finally { globalThis.fetch = original; }
});

test('invalid numeric input is rejected before JSON can silently convert it to unknown', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('must not send invalid input'); };
  try {
    for (const facts of [{ injuredCount: NaN }, { avgSpeed: Infinity }, { vehicleCount: 1.5 },
      { duration: -1 }, { injuredCount: '0' }, { queueLength: 1000001 }]) {
      await assert.rejects(eventIntakeApi.confirm({ facts }), /请核对数值/);
    }
  } finally { globalThis.fetch = original; }
});

test('confirmation summary exposes unknown-to-zero edits and seconds before saving', () => {
  const parsed = { injuredCount: null, fatalityCount: null, duration: null };
  const html = renderToStaticMarkup(React.createElement(IntakeConfirmationSummary,
    { parsedFacts: parsed, facts: { ...parsed, injuredCount: 0, duration: 5 } }));
  assert.match(html, /受伤人数：0；死亡人数：未知；持续时间：5 秒/);
  assert.match(html, /受伤人数：未知 → 0/);
  assert.match(html, /持续时间（秒）：未知 → 5/);
});
