"""A first read of an empty private Case Memory DB must be valid retrieval."""
def test_first_case_search_initializes_empty_database(tmp_path,monkeypatch):
    from backend.case_memory import repository as repo
    monkeypatch.setattr(repo._config,"DB_PATH",str(tmp_path/"fresh.db"))
    result=repo.SQLiteCaseMemoryRepository().find_context_candidates(
        region_id="new-region",event_type="accident",road_id="new-road",intersection_id=None,
        as_of="2026-11-09T09:00:00+08:00",limit=3,exclude_event_id="current-event")
    assert result["cases"]==[] and result["total"]==0
