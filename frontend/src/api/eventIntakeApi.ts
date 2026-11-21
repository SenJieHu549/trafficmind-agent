const API = '/api';

export interface IntakeLocation {
  name: string;
  status: string;
  verificationStatus?: string;
  metadata?: { dataReality?: string };
}

export function isDemoLocation(location?: IntakeLocation): boolean {
  return location?.verificationStatus === 'synthetic' || location?.metadata?.dataReality === 'synthetic';
}

export interface IntakeFacts {
  eventType: string;
  vehicleCount: number | null;
  injuredCount: number | null;
  fatalityCount: number | null;
  avgSpeed: number | null;
  queueLength: number | null;
  duration: number | null;
  isMainRoad: boolean;
  nearbySchool: boolean;
  nearbyHospital: boolean;
  weather: string;
  timePeriod: string;
}
export interface IntakeDraft {
  description: string;
  facts: IntakeFacts;
  suggestedRoadName: string;
  unknownFields: string[];
  warnings: string[];
}
export interface IntakeConfirmation {
  requestId: string;
  description: string;
  confirmed: boolean;
  regionId: string;
  roadId: string;
  facts: IntakeFacts;
  dataReality: 'user_reported' | 'synthetic';
}
export interface IntakeResult {
  eventId: string;
  duplicate: boolean;
  analysisStarted: boolean;
  next: { analysisUrl: string; eventUrl: string };
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg || '字段不合法').join('；')
      : typeof detail === 'string' ? detail : detail?.message;
    throw new Error(message || `请求失败（HTTP ${response.status}）`);
  }
  return data as T;
}

export const eventIntakeApi = {
  preview: (description: string) => request<IntakeDraft>('/events/intake/preview', { description }),
  confirm: async (body: IntakeConfirmation) => {
    // JSON.stringify silently converts NaN/Infinity to null. Reject these before
    // transport so invalid numeric input never masquerades as an unknown value.
    const limits = { vehicleCount: 100000, injuredCount: 100000, fatalityCount: 100000,
      avgSpeed: 500, queueLength: 1000000, duration: 31536000 };
    for (const [key, max] of Object.entries(limits)) {
      const value = body.facts?.[key as keyof typeof limits];
      if (value == null) continue;
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > max ||
          (key.endsWith('Count') && !Number.isInteger(value))) {
        throw new Error('请核对数值：人数和车辆数须为非负整数，观测值须在有效范围内；未知请留空。');
      }
    }
    return request<IntakeResult>('/events/intake/confirm', body);
  },
  regions: () => request<{ regions: (IntakeLocation & { regionId: string })[] }>('/regional/regions'),
  roads: (regionId: string) => request<{ roads: (IntakeLocation & { roadId: string })[] }>(`/regional/regions/${encodeURIComponent(regionId)}/roads`),
};
