interface EventIdentityInput {
  roadName?: string | null;
  eventType?: string | null;
  eventTypeCn?: string | null;
}

const EVENT_TYPE_LABELS: Record<string, string> = {
  accident: '交通事故',
  congestion: '拥堵',
  construction: '施工',
  vehicle_stopped: '车辆滞留',
  illegal_parking: '违停',
  parking_violation: '违停',
  violation: '违停',
  事故: '交通事故',
  拥堵: '拥堵',
  施工: '施工',
  违停: '违停',
};

function cleanText(value?: string | null): string {
  return typeof value === 'string' ? value.trim() : '';
}

// Presentation aliases for the two legacy walkthrough labels. Never use these
// values for API requests, location binding, identity comparisons or snapshots.
export function locationDisplayName(value?: string | null): string {
  const name = cleanText(value);
  if (name === '手动流程验证区（虚构）') return '校园周边片区';
  if (name === '手动验证学校门口路') return '学校门前路';
  if (name === '手动验证学校（虚构）') return '校园小学';
  return name;
}

export function eventTypeLabel(value?: string | null): string {
  const clean = cleanText(value);
  if (/^(evt_|event_|E_[A-Z0-9_]+$|E\d{6,})/.test(clean)) return '';
  return clean ? (EVENT_TYPE_LABELS[clean] || clean) : '';
}

export function eventTitle(input: EventIdentityInput, fallback = '交通事件'): string {
  const road = locationDisplayName(input.roadName);
  const type = eventTypeLabel(input.eventTypeCn || input.eventType);
  if (road && type) return `${road}${type}`;
  if (type) return type;
  if (road) return road;
  return fallback;
}

export function isIncompleteEvent(input: EventIdentityInput): boolean {
  return !cleanText(input.roadName) || !eventTypeLabel(input.eventTypeCn || input.eventType);
}

export function planVersionLabel(version?: number | null, replanCount?: number | null): string {
  if (typeof replanCount === 'number' && replanCount > 0) return `第 ${replanCount} 次调整`;
  if (!version || version <= 1) return '初始方案';
  return `方案版本 ${version}`;
}

export function knowledgeVersionLabel(version?: number | null): string {
  return `版本 ${version && version > 0 ? version : 1}`;
}

export function workflowTemplateVersionLabel(version?: number | null): string {
  return `模板版本 ${version && version > 0 ? version : 1}`;
}
