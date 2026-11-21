import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { loadTs } from './loadTs.mjs';

const { shuBaoshanGeography: geo } = loadTs(new URL('../src/data/regions/shuBaoshanGeography.ts', import.meta.url));
const { DEFAULT_MAP_REGION, mapRegionForProfile } = loadTs(new URL('../src/data/regions/mapRegions.ts', import.meta.url));
const { locateEvent, pilotBounds } = loadTs(new URL('../src/components/map/mapLocation.ts', import.meta.url));
const pack = JSON.parse(readFileSync(new URL('../../backend/data/pilot_regions/shu_baoshan_001/intersections.json', import.meta.url), 'utf8'));

test('default map is SHU Baoshan while the explicit Qiantang profile is preserved', () => {
  assert.equal(DEFAULT_MAP_REGION.geography.metadata.regionId, 'SHU_BAOSHAN_001');
  assert.equal(mapRegionForProfile().title, '上海大学宝山校区周边');
  assert.equal(mapRegionForProfile('qiantang-demo').geography.metadata.regionId, 'QT_BY_XIASHA_PILOT_001');
});

test('SHU map coordinates exactly match imported inventory and preserve OSM provenance', () => {
  assert.equal(geo.features.length, 7);
  assert.equal(new Set(geo.features.map(f => f.id)).size, 7);
  for (const feature of geo.features) {
    const row = pack.find(p => p.intersectionId === feature.id);
    assert.ok(row);
    assert.deepEqual(feature.geometry.coordinates, [row.longitude, row.latitude]);
    assert.deepEqual(feature.properties.osmNodeIds, row.metadata.osmNodeIds);
    assert.equal(feature.properties.verifiedAt, row.verifiedAt);
  }
  assert.equal(geo.metadata.license, 'ODbL-1.0');
  assert.match(geo.metadata.source.sourceUrl, /openstreetmap.org\/api/);
});

test('default viewport fits only source-backed points near SHU Baoshan', () => {
  const bounds = pilotBounds();
  assert.ok(bounds[0][0] > 121.38 && bounds[1][0] < 121.41);
  assert.ok(bounds[0][1] > 31.30 && bounds[1][1] < 31.33);
  for (const feature of geo.features) {
    const [x, y] = feature.geometry.coordinates;
    assert.ok(x >= bounds[0][0] && x <= bounds[1][0] && y >= bounds[0][1] && y <= bounds[1][1]);
  }
});

test('SHU event markers require exact event, region, active binding and matching coordinates', () => {
  const payload = { binding: { eventId: 'E1', bindingId: 'b1', regionId: 'SHU_BAOSHAN_001',
    intersectionId: pack[0].intersectionId, status: 'resolved', resolvedAt: '2026-11-11' }, locationContext: { intersection: pack[0] } };
  assert.deepEqual(locateEvent('E1', payload).coordinates, geo.features[0].geometry.coordinates);
  assert.equal(locateEvent('E1', { ...payload, binding: { ...payload.binding, regionId: 'QT_BY_XIASHA_PILOT_001' } }), null);
  assert.equal(locateEvent('E1', { ...payload, locationContext: { intersection: { ...pack[0], longitude: 120 } } }), null);
  assert.equal(locateEvent('E1', { ...payload, binding: { ...payload.binding, intersectionId: null, roadId: 'SHU_BS_RD_SHANGDA' } }), null);
});

test('map heading, viewport and point layer use the same region profile', () => {
  const panel = readFileSync(new URL('../src/components/map/PilotMapPanel.tsx', import.meta.url), 'utf8');
  const view = readFileSync(new URL('../src/components/map/RealMapView.tsx', import.meta.url), 'utf8');
  assert.match(panel, /region.title/);
  assert.match(panel, /region.geography.metadata.regionId/);
  assert.match(view, /pilotBounds\(region.geography\)/);
  assert.match(view, /data: region.geography/);
  assert.ok(!view.includes('钱塘 Pilot 公开地图'));
});
