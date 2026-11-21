"""Source-backed SHU geometry and real regional import/binding contracts."""
import json
from pathlib import Path
import sqlite3

import pytest

from backend.regional.importer import load_context_pack_from_directory
from backend.regional.repository import SQLiteRegionalRepository
from backend.regional.resolver import EventLocationResolver, EventLocationBindingService

PACK = Path(__file__).resolve().parents[1] / 'data/pilot_regions/shu_baoshan_001'
REGION = 'SHU_BAOSHAN_001'


@pytest.fixture
def imported(tmp_path):
    repo = SQLiteRegionalRepository(str(tmp_path / 'regional.db'))
    pack = load_context_pack_from_directory(PACK)
    repo.import_context_pack(pack)
    return repo, pack


def test_coordinates_and_connectivity_are_backed_by_exact_osm_nodes():
    pack = load_context_pack_from_directory(PACK)
    raw = json.loads((PACK / 'source_snapshot/osm_roads.json').read_text(encoding='utf-8'))
    roads = {r['roadId']: r for r in pack['roads']}
    assert len(roads) == 6 and len(pack['intersections']) == 7 and len(pack['roadRelations']) == 14
    for intersection in pack['intersections']:
        node_id = intersection['metadata']['coordinateNodeId']
        source = raw['nodes'][node_id]
        assert [intersection['longitude'], intersection['latitude']] == [source['longitude'], source['latitude']]
        assert 121.38 <= intersection['longitude'] <= 121.41 and 31.305 <= intersection['latitude'] <= 31.330
        for rid in intersection['metadata']['connectedRoadIds']:
            assert any(way['name'] == roads[rid]['name'] and int(node_id) in way['nodes'] for way in raw['ways'])
        assert intersection['verificationStatus'] == 'verified'
    for poi in pack['pois']:
        assert poi['roadId'] in roads and poi['sourceType'] == 'first_party'
        assert not poi.get('latitude') and not poi.get('longitude')


def test_import_is_idempotent_and_does_not_create_events(imported):
    repo, pack = imported
    assert repo.import_context_pack(pack)['totalNewRecords'] == 0
    with sqlite3.connect(repo.db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'event_records' not in tables and 'workflow_runs' not in tables


def test_every_road_and_intersection_alias_resolves_uniquely(imported):
    repo, pack = imported
    resolver = EventLocationResolver(repo)
    for field, collection in [('roadId', 'roads'), ('intersectionId', 'intersections')]:
        for row in pack[collection]:
            for name in [row['name'], *row.get('aliases', [])]:
                result = resolver.resolve({'roadName': name}, region_id=REGION)
                assert result['status'] == 'resolved' and result[field] == row[field], result
    assert resolver.resolve({'roadName': '不存在的道路'}, region_id=REGION)['status'] == 'unresolved'


def test_bound_event_queries_real_campus_poi_instead_of_guessing_from_text(imported, monkeypatch):
    from backend import config
    from backend.tools import db_tools
    from backend.event_ingestion import ingest_event
    repo, _ = imported
    monkeypatch.setattr(config, 'DB_PATH', repo.db_path)
    monkeypatch.setattr(db_tools, 'DB_PATH', repo.db_path)
    result = ingest_event(source='isolated_test', source_event_id='SHU-fixture-1', event={
        'eventType': 'accident', 'roadName': '上大路', 'avgSpeed': 10, 'queueLength': 20,
        'duration': 60, 'confidence': 0.9, 'vehicleCount': 2})
    bound = EventLocationBindingService(repo).resolve_and_bind(result['eventId'], region_id=REGION)
    assert bound['binding']['roadId'] == 'SHU_BS_RD_SHANGDA'
    context = repo.build_regional_location_context(bound['binding'])
    campus = next(poi for poi in context['nearbyPois'] if poi['type'] == 'school')
    assert campus['name'] == '上海大学宝山校区'
    assert campus['metadata']['address'] == '上海市宝山区上大路99号'
    assert campus['verificationStatus'] == 'verified'
