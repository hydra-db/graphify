"""Live integration test for the HydraDB managed platform.

Exercises the full documented lifecycle against api.hydradb.com:
provision database -> wait ready -> ingest knowledge -> wait for indexing ->
hybrid query with graph context -> feedback with the query's request_id ->
delete the ingested context.

Auto-skips unless HYDRA_DB_API_KEY is set, so it is a no-op in default CI.
It uses (and provisions on first run) a dedicated database named
``graphify-e2e`` and cleans up the sources it ingested; the database itself
is left in place because provisioning is the expensive step and re-creating
it every run would be wasteful.

    HYDRA_DB_API_KEY=sk_... uv run pytest tests/test_hydradb_cloud_integration.py -q
"""
from __future__ import annotations

import os
import uuid

import pytest

from graphify.hydradb_cloud import (
    API_KEY_ENV,
    HydraDBCloudClient,
    HydraDBCloudError,
    build_llm_context,
    format_query_result,
    sync_out_dir,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get(API_KEY_ENV),
    reason=f"{API_KEY_ENV} not set - live HydraDB test skipped",
)

DATABASE = os.environ.get("HYDRADB_E2E_DATABASE", "graphify-e2e")
# A unique marker per run proves retrieval hits *this* run's document, not a
# leftover from an earlier one.
MARKER = f"marker-{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="module")
def client():
    return HydraDBCloudClient()


@pytest.fixture(scope="module")
def synced(client, tmp_path_factory):
    out = tmp_path_factory.mktemp("hydra-out")
    (out / "GRAPH_REPORT.md").write_text(
        "# Graph report: integration fixture\n\n"
        f"The secret integration marker is {MARKER}. "
        "MultiHeadAttention is implemented in attention.py and used by the "
        "TransformerBlock in transformer.py.\n",
        encoding="utf-8",
    )
    if DATABASE not in client.list_databases():
        client.create_database(DATABASE)
    client.wait_until_ready(DATABASE, timeout=300)
    summary = sync_out_dir(client, out, DATABASE, log=lambda *_: None)
    yield summary
    if summary["ids"]:
        client.delete_context(DATABASE, summary["ids"])


def test_sync_ingests_and_indexes(synced):
    assert synced["database"] == DATABASE
    assert synced["ids"], "ingest returned no source ids"
    assert not synced["failed"]
    assert all(
        s.get("indexing_status") == "completed" for s in synced["statuses"]
    )


def test_query_retrieves_the_synced_document(client, synced):
    data = client.query(
        DATABASE,
        f"What is the secret integration marker? {MARKER}",
        max_results=3,
    )
    chunks = data.get("chunks") or []
    assert chunks, "query returned no chunks"
    assert any(MARKER in (c.get("chunk_content") or "") for c in chunks), (
        "the synced document was not retrieved"
    )
    assert data.get("request_id")
    # the renderer must handle the live payload end to end
    assert MARKER in format_query_result(data)


def test_llm_context_uses_sdk_build_string(synced):
    pytest.importorskip("hydra_db", reason="needs the hydradb-sdk extra")
    context, request_id = build_llm_context(
        DATABASE, f"What is the secret integration marker? {MARKER}",
        max_results=3,
    )
    assert MARKER in context
    assert request_id
    # the SDK path's request_id must work with this module's own feedback()
    result = HydraDBCloudClient().feedback(
        DATABASE, request_id, feedback="llm-context integration test"
    )
    assert result.get("recorded") is True


def test_llm_context_wraps_not_found_cleanly():
    pytest.importorskip("hydra_db", reason="needs the hydradb-sdk extra")
    with pytest.raises(HydraDBCloudError, match="not found"):
        build_llm_context("graphify-e2e-no-such-database", "q")


def test_feedback_round_trips_with_request_id(client, synced):
    data = client.query(DATABASE, "What uses MultiHeadAttention?",
                        max_results=1)
    result = client.feedback(
        DATABASE,
        data["request_id"],
        feedback="integration test: retrieval quality acceptable",
        metadata={"agent": "graphify-integration-test"},
    )
    assert result.get("recorded") is True
    assert result.get("request_id") == data["request_id"]


def test_database_status_reports_ready(client, synced):
    infra = client.database_status(DATABASE).get("infra") or {}
    assert infra.get("ready_for_ingestion") is True
