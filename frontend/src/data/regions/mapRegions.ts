import { shuBaoshanGeography } from './shuBaoshanGeography';
import { pilotGeography } from '../pilot/pilotGeography';

export interface RegionGeography {
  type: string;
  metadata: { regionId: string };
  features: Array<{ type: string; id: string; geometry: { type: string; coordinates: number[] };
    properties: { name: string; verificationStatus: string; coordinateSystem: string } }>;
}
export interface MapRegion { title: string; subtitle: string; geography: RegionGeography }
export const SHU_MAP_REGION: MapRegion = {
  title: '上海大学宝山校区周边', subtitle: '上海 · 宝山 · 上大路—锦秋路—南陈路', geography: shuBaoshanGeography,
};
export const QIANTANG_MAP_REGION: MapRegion = {
  title: '钱塘区域地图', subtitle: '钱塘 · 白杨－金沙湖－下沙高教园', geography: pilotGeography,
};
export const DEFAULT_MAP_REGION = SHU_MAP_REGION;
export function mapRegionForProfile(profile?: string): MapRegion {
  return profile === 'qiantang-demo' ? QIANTANG_MAP_REGION : DEFAULT_MAP_REGION;
}
