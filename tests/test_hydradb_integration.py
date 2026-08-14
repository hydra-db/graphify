"""Integration test for push_to_hydradb against a real HydraDB node.

Runs for real against a local development node (see the hydra-db/hydradb
README, "Run with Docker" / "Run a local server"):

    mkdir -p hydradb-data/store hydradb-data/cache
    printf '%s\n' 'local-development-token-32-bytes' > hydradb-data/auth-token
    docker run --rm --user "$(id -u):$(id -g)" \
      -p 7687:7687 -p 8443:8443 -p 9090:9090 \
      -v "$PWD/hydradb-data:/data" \
      -e CLOUD_PROVIDER=local -e LOCAL_PATH=/data/store \
      -e GRAPH_NAMESPACE=default -e GRAPH_ID=default \
      -e GRAPH_CELL_ID=cell-0 -e GRAPH_CELLS=cell-0 -e GRAPH_NODE_ID=node-0 \
      -e GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 \
      -e GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687 \
      -e GRAPH_DATA_CACHE_DIR=/data/cache \
      -e GRAPH_AUTH_TOKEN_FILE=/data/auth-token \
      -e GRAPH_ALLOW_PLAINTEXT=true -e RUST_MIN_STACK=33554432 \
      ghcr.io/hydra-db/hydradb:latest
    uv run pytest tests/test_hydradb_integration.py -q

The test auto-skips when the `neo4j` driver is not installed or no HydraDB is
reachable, so it is a no-op in the default CI (which runs no external
services). Connection is overridable via HYDRADB_HOST / HYDRADB_PORT /
HYDRADB_TOKEN.

Note: HydraDB has no DROP/graph-delete over Bolt, and a bare `MATCH (n)`
needs a predicate, so assertions count by label instead of assuming an empty
store; the pushed vertices are found via their stable hash ids.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

neo4j = pytest.importorskip("neo4j")

FIXTURES = Path(__file__).parent / "fixtures"
HOST = os.environ.get("HYDRADB_HOST", "localhost")
PORT = int(os.environ.get("HYDRADB_PORT", "7687"))
TOKEN = os.environ.get("HYDRADB_TOKEN", "local-development-token-32-bytes")
URI = f"bolt://{HOST}:{PORT}"


def _connect():
    """Return a connected Bolt driver, or skip if no HydraDB is reachable."""
    try:
        driver = neo4j.GraphDatabase.driver(
            URI, auth=("neo4j", TOKEN), connection_timeout=3
        )
        driver.verify_connectivity()
        return driver
    except Exception as e:  # pragma: no cover - depends on local environment
        pytest.skip(f"no HydraDB reachable at {URI} ({e})")


@pytest.fixture()
def driver():
    d = _connect()
    yield d
    d.close()


def _fixture_graph():
    from graphify.build import build_from_json

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    return build_from_json(extraction)


def _count_vertices(session, G) -> int:
    """Count how many of G's nodes exist in the store, by stable hash id."""
    from graphify.exporters.hydradb import hydradb_vertex_id

    found = 0
    for node_id in G.nodes():
        vid = hydradb_vertex_id(str(node_id))
        rec = session.run(
            "MATCH (n {id: $vid}) RETURN count(*) AS c", vid=vid
        ).single()
        found += int(rec["c"])
    return found


def _count_edges(session, G) -> int:
    from graphify.exporters.hydradb import hydradb_statements

    found = 0
    for cypher, params in hydradb_statements(G):
        if "MERGE (s)-[r:" not in cypher:
            continue
        rel = cypher.split("MERGE (s)-[r:")[1].split(" ")[0]
        for row in params["rows"]:
            rec = session.run(
                f"MATCH (s {{id: $src}})-[r:{rel}]->(d {{id: $dst}}) "
                f"RETURN count(*) AS c",
                src=row["source_vertex"],
                dst=row["destination_vertex"],
            ).single()
            found += int(rec["c"])
    return found


def test_push_to_hydradb_round_trips(driver):
    from graphify.export import push_to_hydradb

    G = _fixture_graph()
    result = push_to_hydradb(G, uri=URI, password=TOKEN)

    assert result["nodes"] == G.number_of_nodes()
    assert result["edges"] == G.number_of_edges()

    with driver.session() as session:
        assert _count_vertices(session, G) == G.number_of_nodes()
        assert _count_edges(session, G) >= G.number_of_edges()


def test_push_to_hydradb_is_idempotent(driver):
    """MERGE-based push is safe to re-run - counts must not grow."""
    from graphify.export import push_to_hydradb

    G = _fixture_graph()
    push_to_hydradb(G, uri=URI, password=TOKEN)
    before_edges = None
    with driver.session() as session:
        before_edges = _count_edges(session, G)

    push_to_hydradb(G, uri=URI, password=TOKEN)
    with driver.session() as session:
        assert _count_vertices(session, G) == G.number_of_nodes()
        assert _count_edges(session, G) == before_edges


def test_pushed_properties_survive_read_back(driver):
    from graphify.export import push_to_hydradb
    from graphify.exporters.hydradb import hydradb_vertex_id

    G = _fixture_graph()
    push_to_hydradb(G, uri=URI, password=TOKEN)

    some_node = next(iter(G.nodes()))
    vid = hydradb_vertex_id(str(some_node))
    with driver.session() as session:
        rec = session.run(
            "MATCH (n {id: $vid}) RETURN n.key AS key", vid=vid
        ).single()
    assert rec["key"] == str(some_node)
