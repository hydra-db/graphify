"""Integration test for push_to_hydradb against a real HydraDB node.

Runs for real against a local single-node HydraDB (see the HydraDB README's
"Run with Docker" section):

    docker run --rm -p 7687:7687 ... ghcr.io/hydra-db/hydradb:latest
    uv run pytest tests/test_hydradb_integration.py -q

The test auto-skips when the `neo4j` driver is not installed or no HydraDB is
reachable, so it is a no-op in the default CI (which runs no external
services). Connection is overridable via HYDRADB_URI / HYDRADB_TOKEN /
HYDRADB_DATABASE; the defaults match the README's local dev flow.

HydraDB stores one graph per node (no named graphs to drop), so instead of
deleting a scratch graph the fixture DETACH DELETEs exactly the vertex ids the
exporter derives for the fixture graph, before and after each test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

neo4j = pytest.importorskip("neo4j")

FIXTURES = Path(__file__).parent / "fixtures"
URI = os.environ.get("HYDRADB_URI", "bolt://localhost:7687")
TOKEN = os.environ.get("HYDRADB_TOKEN", "local-development-token-32-bytes")
DATABASE = os.environ.get("HYDRADB_DATABASE", "default")


def _fixture_graph():
    from graphify.build import build_from_json

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    return build_from_json(extraction)


def _connect():
    """Return a connected Bolt driver, or skip if no HydraDB is reachable."""
    try:
        driver = neo4j.GraphDatabase.driver(URI, auth=("neo4j", TOKEN))
        driver.verify_connectivity()
        return driver
    except Exception as e:  # pragma: no cover - depends on local environment
        pytest.skip(f"no HydraDB reachable at {URI} ({e})")


def _delete_fixture_vertices(driver, G):
    """Remove exactly the vertices the exporter would create for G."""
    from graphify.exporters.graphdb import _hydradb_id

    rows = [{"vertex": _hydradb_id(str(n))} for n in G.nodes]
    with driver.session(database=DATABASE) as session:
        session.run(
            "UNWIND $vertices AS row MATCH (n {id: row.vertex}) DETACH DELETE n",
            vertices=rows,
        ).consume()


def _count_fixture_vertices(driver, G):
    """Count how many of G's derived vertex ids exist on the server."""
    from graphify.exporters.graphdb import _hydradb_id

    found = 0
    with driver.session(database=DATABASE) as session:
        for n in G.nodes:
            record = session.run(
                "MATCH (v:Entity {id: $id}) RETURN count(*) AS c",
                id=_hydradb_id(str(n)),
            ).single()
            found += record["c"]
    return found


@pytest.fixture()
def db():
    driver = _connect()
    G = _fixture_graph()
    _delete_fixture_vertices(driver, G)
    yield driver
    _delete_fixture_vertices(driver, G)
    driver.close()


def test_push_to_hydradb_creates_expected_graph(db):
    from graphify.export import push_to_hydradb

    G = _fixture_graph()
    result = push_to_hydradb(G, uri=URI, password=TOKEN, database=DATABASE)

    assert result["nodes"] == G.number_of_nodes()
    assert result["edges"] == G.number_of_edges()
    assert _count_fixture_vertices(db, G) == G.number_of_nodes()

    # The original string ids survive as the uid property, and edges are
    # typed from the relation attribute and traversable between them.
    some_node = next(iter(G.nodes))
    from graphify.exporters.graphdb import _hydradb_id

    with db.session(database=DATABASE) as session:
        record = session.run(
            "MATCH (n:Entity {id: $id}) RETURN n.uid AS uid",
            id=_hydradb_id(str(some_node)),
        ).single()
        assert record["uid"] == str(some_node)

        u, v, data = next(iter(G.edges(data=True)))
        rel = data.get("relation", "RELATED_TO")
        safe_rel = "".join(c if c.isalnum() or c == "_" else "_" for c in rel.upper()) or "RELATED_TO"
        record = session.run(
            f"MATCH (a:Entity {{id: $src}})-[r:{safe_rel}]->(b:Entity {{id: $dst}}) "
            "RETURN count(*) AS c",
            src=_hydradb_id(str(u)),
            dst=_hydradb_id(str(v)),
        ).single()
        assert record["c"] == 1


def test_push_to_hydradb_is_idempotent(db):
    """MERGE-based push is safe to re-run - counts must not grow."""
    from graphify.export import push_to_hydradb

    G = _fixture_graph()
    push_to_hydradb(G, uri=URI, password=TOKEN, database=DATABASE)
    push_to_hydradb(G, uri=URI, password=TOKEN, database=DATABASE)

    assert _count_fixture_vertices(db, G) == G.number_of_nodes()
