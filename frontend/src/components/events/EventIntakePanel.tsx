import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { eventIntakeApi, isDemoLocation, type IntakeLocation, type IntakeDraft, type IntakeFacts, type IntakeResult } from '../../api/eventIntakeApi';
import { locationDisplayName } from '../../utils/display';

const fieldStyle: CSSProperties = { display: 'grid', gap: 5, fontSize: 13, color: '#334155' };
const inputStyle: CSSProperties = { padding: '8px 10px', border: '1px solid #CBD5E1', borderRadius: 6, minWidth: 0, width: '100%', boxSizing: 'border-box', background: '#FFF' };
const buttonStyle: CSSProperties = { padding: '9px 14px', border: 0, borderRadius: 7, background: '#0F766E', color: '#FFF', cursor: 'pointer' };
const types = [['accident', '交通事故'], ['congestion', '拥堵'], ['illegal_parking', '违停'], ['wrong_way', '逆行'],
  ['pedestrian_intrusion', '行人闯入'], ['signal_fault', '信号灯故障'], ['vehicle_stopped', '车辆停驶'], ['construction_block', '施工占道']];
const numericFields: [keyof IntakeFacts, string, boolean][] = [
  ['vehicleCount', '涉事车辆数', true], ['injuredCount', '受伤人数', true], ['fatalityCount', '死亡人数', true],
  ['avgSpeed', '平均车速（km/h）', false], ['queueLength', '排队长度（米）', false], ['duration', '持续时间（秒）', false],
];

export function IntakeFactsFields({ facts, onChange, disabled = false }: {
  facts: IntakeFacts; onChange: (facts: IntakeFacts) => void; disabled?: boolean;
}) {
  return <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12 }}>
    <label style={fieldStyle}>事件类型<select style={inputStyle} value={facts.eventType} disabled={disabled}
      onChange={e => onChange({ ...facts, eventType: e.target.value })}>
      <option value="">请确认事件类型</option>{types.map(([value, name]) => <option key={value} value={value}>{name}</option>)}
    </select></label>
    {numericFields.map(([key, label, integer]) => <label key={key} style={fieldStyle}>{label}
      <input type="number" min={0} step={integer ? 1 : 'any'} style={inputStyle} disabled={disabled}
        value={facts[key] == null ? '' : String(facts[key])} placeholder="未知（可留空）"
        onChange={e => onChange({ ...facts, [key]: e.target.value === '' ? null : Number(e.target.value) })} />
    </label>)}
    <label style={fieldStyle}>天气<select style={inputStyle} value={facts.weather} disabled={disabled}
      onChange={e => onChange({ ...facts, weather: e.target.value })}>
      {[['unknown', '未知'], ['clear', '晴'], ['rain', '雨'], ['snow', '雪'], ['fog', '雾']].map(([v, n]) => <option key={v} value={v}>{n}</option>)}
    </select></label>
    <label style={fieldStyle}>时段<select style={inputStyle} value={facts.timePeriod} disabled={disabled}
      onChange={e => onChange({ ...facts, timePeriod: e.target.value })}>
      {[['unknown', '未知'], ['off_peak', '非高峰'], ['morning_peak', '早高峰'], ['evening_peak', '晚高峰']].map(([v, n]) => <option key={v} value={v}>{n}</option>)}
    </select></label>
    <label style={{ ...fieldStyle, display: 'flex', alignItems: 'center' }}><input type="checkbox" checked={facts.isMainRoad ?? false} disabled={disabled}
      onChange={e => onChange({ ...facts, isMainRoad: e.target.checked })} />登记为主干道</label>
    <label style={{ ...fieldStyle, display: 'flex', alignItems: 'center' }}><input type="checkbox" checked={facts.nearbySchool} disabled={disabled}
      onChange={e => onChange({ ...facts, nearbySchool: e.target.checked })} />登记为学校附近</label>
    <label style={{ ...fieldStyle, display: 'flex', alignItems: 'center' }}><input type="checkbox" checked={facts.nearbyHospital} disabled={disabled}
      onChange={e => onChange({ ...facts, nearbyHospital: e.target.checked })} />登记为医院附近</label>
  </div>;
}

export function IntakeConfirmationSummary({ facts, parsedFacts }: { facts: IntakeFacts; parsedFacts: IntakeFacts | null }) {
  const labels: Record<keyof IntakeFacts, string> = { eventType: '事件类型', vehicleCount: '涉事车辆数',
    injuredCount: '受伤人数', fatalityCount: '死亡人数', avgSpeed: '平均车速', queueLength: '排队长度',
    duration: '持续时间（秒）', isMainRoad: '主干道标记', nearbySchool: '学校附近标记', nearbyHospital: '医院附近标记',
    weather: '天气', timePeriod: '时段' };
  const text = (value: unknown) => value == null || value === 'unknown' ? '未知'
    : typeof value === 'boolean' ? (value ? '已登记' : '未登记')
    : [...types, ['clear', '晴'], ['rain', '雨'], ['snow', '雪'], ['fog', '雾'],
      ['off_peak', '非高峰'], ['morning_peak', '早高峰'], ['evening_peak', '晚高峰']].find(([v]) => v === value)?.[1] || String(value);
  const changes = parsedFacts ? (Object.keys(labels) as (keyof IntakeFacts)[])
    .filter(key => facts[key] !== parsedFacts[key]) : [];
  return <section aria-label="登记核对摘要" style={{ background: '#F8FAFC', padding: 12, borderRadius: 6, fontSize: 13 }}>
    <strong>请核对最终登记内容</strong>
    <p style={{ margin: '8px 0 0' }}>受伤人数：{text(facts.injuredCount)}；死亡人数：{text(facts.fatalityCount)}；持续时间：{text(facts.duration)}{facts.duration == null ? '' : ' 秒'}。</p>
    {changes.length > 0 && <><p style={{ margin: '8px 0 0' }}>以下字段已调整，请确认与最新上报信息一致：</p>
      <ul style={{ margin: '6px 0 0', paddingLeft: 20 }}>{changes.map(key => <li key={key}>{labels[key]}：{text(parsedFacts?.[key])} → {text(facts[key])}</li>)}</ul>
    </>}
  </section>;
}

export default function EventIntakePanel({ defaultOpen = false, onCreated }: {
  defaultOpen?: boolean; onCreated?: (eventId: string) => void;
}) {
  const [description, setDescription] = useState('');
  const [draft, setDraft] = useState<IntakeDraft | null>(null);
  const [parsedFacts, setParsedFacts] = useState<IntakeFacts | null>(null);
  const [regions, setRegions] = useState<(IntakeLocation & { regionId: string })[]>([]);
  const [roads, setRoads] = useState<(IntakeLocation & { roadId: string })[]>([]);
  const [regionId, setRegionId] = useState('');
  const [roadId, setRoadId] = useState('');
  const [roadsLoading, setRoadsLoading] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [dataReality, setReality] = useState<'user_reported' | 'synthetic'>('user_reported');
  const [requestId, setRequestId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [created, setCreated] = useState<IntakeResult | null>(null);
  const inFlight = useRef(false);
  const revision = useRef(0);
  const demoLocation = isDemoLocation(regions.find(r => r.regionId === regionId))
    || isDemoLocation(roads.find(r => r.roadId === roadId));

  useEffect(() => {
    let active = true;
    eventIntakeApi.regions().then(data => { if (active) setRegions(data.regions.filter(r => r.status === 'active')); })
      .catch(e => { if (active) setError(`区域列表读取失败：${e.message}`); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    let active = true;
    setRoads([]); setRoadId('');
    if (!regionId) { setRoadsLoading(false); return; }
    setRoadsLoading(true);
    eventIntakeApi.roads(regionId).then(data => {
      if (!active) return;
      const available = data.roads.filter(r => r.status === 'active');
      setRoads(available);
    }).catch(e => { if (active) setError(`道路列表读取失败：${e.message}`); })
      .finally(() => { if (active) setRoadsLoading(false); });
    return () => { active = false; };
  }, [regionId]);

  const changed = () => { setConfirmed(false); setError(''); setRequestId(crypto.randomUUID()); };
  const preview = async () => {
    if (inFlight.current || !description.trim()) return;
    inFlight.current = true; setBusy(true); setError('');
    const at = ++revision.current;
    try {
      const next = await eventIntakeApi.preview(description.trim());
      if (revision.current !== at) return;
      setDraft(next); setParsedFacts({ ...next.facts }); setConfirmed(false); setRequestId(crypto.randomUUID());
    } catch (e) { if (revision.current === at) setError(e instanceof Error ? e.message : '草稿解析失败'); }
    finally { inFlight.current = false; setBusy(false); }
  };
  const save = async () => {
    if (!draft || !confirmed || !regionId || !roadId || !draft.facts.eventType || inFlight.current) return;
    inFlight.current = true; setBusy(true); setError('');
    try {
      const result = await eventIntakeApi.confirm({ requestId, description: draft.description,
        confirmed: true, regionId, roadId, facts: draft.facts, dataReality });
      setCreated(result); onCreated?.(result.eventId);
    } catch (e) { setError(e instanceof Error ? e.message : '登记失败，可重试同一次确认'); }
    finally { inFlight.current = false; setBusy(false); }
  };
  return <details open={defaultOpen || undefined} style={{ border: '1px solid #CBD5E1', borderRadius: 10, padding: 16, background: '#FFF', marginBottom: 14 }}>
    <summary style={{ cursor: 'pointer', fontWeight: 700, color: '#134E4A' }}>事件登记</summary>
    <div style={{ display: 'grid', gap: 12, marginTop: 12 }}>
      <p style={{ margin: 0, color: '#64748B', fontSize: 13 }}>填写事件情况，核对信息和事发道路，确认登记后进入协同研判。</p>
      {error && <p role="alert" style={{ color: '#B91C1C', margin: 0 }}>{error}</p>}
      {created ? <div role="status">
        <strong>事件已登记，等待研判</strong><p>事件编号：{created.eventId}</p>
        <p>原始报告：{draft?.description}</p>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          <a href={created.next.analysisUrl} style={{ color: '#0F766E', fontWeight: 700 }}>进入该事件研判</a>
          <a href={created.next.eventUrl} style={{ color: '#0F766E' }}>在交通态势中查看</a>
          <button type="button" onClick={() => { setCreated(null); setDraft(null); setParsedFacts(null); setDescription(''); setConfirmed(false); setRequestId(''); }}>登记另一事件</button>
        </div>
      </div> : <>
        <label style={fieldStyle}>事件原始描述<textarea value={description} disabled={busy} rows={3} style={inputStyle}
          placeholder="例如：学校附近两车相撞，伤亡人数未知。请在确认卡片中选择具体道路。"
          onChange={e => { revision.current++; setDescription(e.target.value); setDraft(null); setParsedFacts(null); setConfirmed(false); setError(''); }} /></label>
        <div><button type="button" style={buttonStyle} disabled={busy || !description.trim()} onClick={preview}>{busy ? '处理中…' : '解析事件'}</button></div>
        {draft && <section aria-label="待确认事件信息" style={{ display: 'grid', gap: 14 }}>
          <p style={{ margin: 0, fontSize: 13 }}>草稿尚未入库。留空表示未知，0 表示明确报告为零；未登记的周边设施不等于不存在。{draft.suggestedRoadName && ` 文本中识别的道路：${draft.suggestedRoadName}，请核对下方选择。`}</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 12 }}>
            <label style={fieldStyle}>所在区域<select value={regionId} style={inputStyle} disabled={busy}
              onChange={e => { changed(); setRegionId(e.target.value); setRoadId(''); }}>
              <option value="">请选择所在区域</option>{regions.map(r => <option key={r.regionId} value={r.regionId}>{locationDisplayName(r.name)}</option>)}
            </select></label>
            <label style={fieldStyle}>具体道路<select value={roadId} style={inputStyle} disabled={busy || roadsLoading || !regionId}
              onChange={e => { changed(); setRoadId(e.target.value); }}>
              <option value="">{roadsLoading ? '正在读取道路…' : '请选择事发道路'}</option>{roads.map(r => <option key={r.roadId} value={r.roadId}>{locationDisplayName(r.name)}</option>)}
            </select></label>
          </div>
          {regionId && !roadsLoading && roads.length === 0 && <p style={{ margin: 0 }}>该区域尚未配置道路，请联系管理员补充道路资料。</p>}
          <IntakeFactsFields facts={draft.facts} disabled={busy} onChange={facts => { changed(); setDraft({ ...draft, facts }); }} />
          <label style={fieldStyle}>信息来源<select style={inputStyle} value={dataReality} disabled={busy}
            onChange={e => { changed(); setReality(e.target.value as 'user_reported' | 'synthetic'); }}>
            <option value="user_reported">人工上报（待核实）</option><option value="synthetic">演示数据</option>
          </select></label>
          {demoLocation && <p style={{ margin: 0, color: '#64748B', fontSize: 13 }}>位置资料来源：演示数据，尚未核验为实际地点。</p>}
          <IntakeConfirmationSummary facts={draft.facts} parsedFacts={parsedFacts} />
          <label style={{ fontSize: 13 }}><input type="checkbox" checked={confirmed} disabled={busy} onChange={e => setConfirmed(e.target.checked)} /> 我已核对以上登记信息；本次确认不代表处置动作已获批准。</label>
          <div><button type="button" style={buttonStyle} disabled={busy || !confirmed || !regionId || !roadId || !draft.facts.eventType} onClick={save}>确认登记</button></div>
        </section>}
      </>}
    </div>
  </details>;
}
