"""Unit tests for the HydraDB managed-platform client (no network).

The live end-to-end flow (provision -> ingest -> query -> feedback) is
tests/test_hydradb_cloud_integration.py, which needs HYDRA_DB_API_KEY.
These tests stub urllib at the module boundary and verify the wire
behaviour the platform's integration guide specifies: bearer + API-Version
headers, envelope unwrapping, retry only on 429/500/503, multipart ingest
encoding, and the graphify sync flow.
"""
from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from graphify.hydradb_cloud import (
    HydraDBCloudClient,
    HydraDBCloudError,
    _encode_multipart,
    collect_sync_files,
    default_database_name,
    format_query_result,
    sync_out_dir,
)


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._body = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(status: int, payload: dict | None = None,
                headers: dict | None = None):
    import email.message

    msg = email.message.Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError(
        "https://api.hydradb.com/x", status, "err", msg,
        io.BytesIO(json.dumps(payload or {}).encode()),
    )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("HYDRA_DB_API_KEY", "sk_test_dummy")
    c = HydraDBCloudClient(backoff=0.0)
    return c


def _patch_urlopen(monkeypatch, responses: list, calls: list):
    """Queue canned responses; HTTPError instances are raised in turn."""

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_request_sends_bearer_and_api_version(client, monkeypatch):
    calls: list = []
    _patch_urlopen(monkeypatch, [
        _FakeResponse({"success": True, "data": {"databases": ["a"]}}),
    ], calls)
    assert client.list_databases() == ["a"]
    req = calls[0]
    assert req.get_header("Authorization") == "Bearer sk_test_dummy"
    assert req.get_header("Api-version") == "2"


def test_envelope_failure_raises_with_code(client, monkeypatch):
    _patch_urlopen(monkeypatch, [
        _FakeResponse({
            "success": False,
            "error": {"code": "DATABASE_NOT_FOUND", "message": "nope"},
        }),
    ], [])
    with pytest.raises(HydraDBCloudError) as exc:
        client.database_status("missing")
    assert exc.value.code == "DATABASE_NOT_FOUND"
    assert "nope" in str(exc.value)


def test_retries_429_then_succeeds(client, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls: list = []
    _patch_urlopen(monkeypatch, [
        _http_error(429),
        _http_error(503),
        _FakeResponse({"success": True, "data": {"databases": []}}),
    ], calls)
    assert client.list_databases() == []
    assert len(calls) == 3


def test_does_not_retry_400(client, monkeypatch):
    calls: list = []
    _patch_urlopen(monkeypatch, [
        _http_error(400, {
            "success": False,
            "detail": {"error_code": "VALIDATION_ERROR", "message": "bad"},
        }),
    ], calls)
    with pytest.raises(HydraDBCloudError) as exc:
        client.create_database("x")
    assert len(calls) == 1
    assert exc.value.status == 400
    assert exc.value.code == "VALIDATION_ERROR"


def test_retries_exhaust_and_raise(client, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls: list = []
    _patch_urlopen(
        monkeypatch, [_http_error(503, {"success": False}) for _ in range(5)], calls
    )
    with pytest.raises(HydraDBCloudError):
        client.list_databases()
    assert len(calls) == 5  # initial + max_retries(4)


def test_multipart_encoding_carries_fields_and_files():
    body, ctype = _encode_multipart(
        {"database": "db1", "type": "knowledge"},
        [("documents", "report.md", b"# hello")],
    )
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    text = body.decode()
    assert f"--{boundary}\r\n" in text
    assert 'name="database"\r\n\r\ndb1' in text
    assert 'name="documents"; filename="report.md"' in text
    assert "Content-Type: text/markdown" in text
    assert "# hello" in text
    assert text.endswith(f"--{boundary}--\r\n")


def test_ingest_files_builds_multipart(client, monkeypatch, tmp_path):
    calls: list = []
    _patch_urlopen(monkeypatch, [
        _FakeResponse({"success": True, "data": {"results": []}}, status=202),
    ], calls)
    doc = tmp_path / "GRAPH_REPORT.md"
    doc.write_text("# report")
    client.ingest_files("db1", [doc], collection="team")
    req = calls[0]
    assert req.get_full_url().endswith("/context/ingest")
    body = req.data.decode()
    assert 'name="collection"\r\n\r\nteam' in body
    assert 'name="upsert"\r\n\r\ntrue' in body
    assert 'filename="GRAPH_REPORT.md"' in body


def test_query_body_composition(client, monkeypatch):
    calls: list = []
    _patch_urlopen(monkeypatch, [
        _FakeResponse({"success": True, "data": {"chunks": []}}),
    ], calls)
    client.query("db1", "what is the auth flow?", type="all",
                 mode="thinking", max_results=5)
    sent = json.loads(calls[0].data)
    assert sent == {
        "database": "db1",
        "query": "what is the auth flow?",
        "type": "all",
        "query_by": "hybrid",
        "graph_context": True,
        "mode": "thinking",
        "max_results": 5,
    }


def test_feedback_requires_text_or_ground_truth(client):
    with pytest.raises(HydraDBCloudError):
        client.feedback("db1")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("HYDRA_DB_API_KEY", raising=False)
    with pytest.raises(HydraDBCloudError):
        HydraDBCloudClient()


def test_default_database_name_is_slugged():
    assert default_database_name("/tmp/My Repo!") == "graphify-my-repo"
    assert default_database_name("/tmp/graphify") == "graphify-graphify"


def test_collect_sync_files_picks_report_and_wiki(tmp_path):
    out = tmp_path / "graphify-out"
    (out / "wiki").mkdir(parents=True)
    (out / "GRAPH_REPORT.md").write_text("# r")
    (out / "graph.json").write_text("{}")  # not prose - never synced
    (out / "wiki" / "index.md").write_text("# w")
    (out / "wiki" / "community-1.md").write_text("# c1")
    files = [f.name for f in collect_sync_files(out)]
    assert files == ["GRAPH_REPORT.md", "community-1.md", "index.md"]


def test_sync_out_dir_provisions_ingests_and_waits(tmp_path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "GRAPH_REPORT.md").write_text("# report")

    events: list = []

    class _FakeClient:
        def list_databases(self):
            events.append("list")
            return ["other-db"]

        def create_database(self, database):
            events.append(f"create:{database}")
            return {"status": "accepted"}

        def wait_until_ready(self, database):
            events.append(f"ready:{database}")
            return {"infra": {"ready_for_ingestion": True}}

        def ingest_files(self, database, files, collection=None):
            events.append(f"ingest:{len(files)}")
            return {"results": [
                {"id": "HydraDoc1", "filename": "GRAPH_REPORT.md", "error": ""},
            ]}

        def wait_for_indexing(self, database, ids):
            events.append(f"index:{ids}")
            return [{"id": "HydraDoc1", "indexing_status": "completed"}]

    summary = sync_out_dir(_FakeClient(), out, "graphify-proj", log=lambda *_: None)
    assert events == [
        "list", "create:graphify-proj", "ready:graphify-proj",
        "ingest:1", "index:['HydraDoc1']",
    ]
    assert summary["ids"] == ["HydraDoc1"]
    assert summary["failed"] == []


def test_sync_out_dir_requires_artifacts(tmp_path):
    with pytest.raises(HydraDBCloudError):
        sync_out_dir(object(), tmp_path, "db", log=lambda *_: None)


def test_format_query_result_renders_chunks_and_triplets():
    data = {
        "chunks": [{
            "source_title": "GRAPH_REPORT.md",
            "relevancy_score": 0.91,
            "chunk_content": "APIRouter is the hub.",
        }],
        "graph_context": {
            "chunk_relations": [{
                "triplets": [{
                    "source": {"name": "APIRouter"},
                    "relation": {"predicate": "uses"},
                    "target": {"name": "Dependant"},
                }],
            }],
        },
    }
    text = format_query_result(data)
    assert "[1] GRAPH_REPORT.md (score 0.91)" in text
    assert "APIRouter is the hub." in text
    assert "APIRouter --uses--> Dependant" in text
    assert format_query_result({"chunks": []}) == "no results"
