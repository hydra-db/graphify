"""Unit tests for the HydraDB core-adapter exporter.

These verify - without a running HydraDB node - that every statement the
adapter emits stays inside HydraDB's OpenCypher subset as documented in
hydra-db/hydradb `cypher-compat.md`:

  - vertex upserts are `UNWIND $rows AS row MERGE (n {id: row.vertex})`
    followed by a fixed per-property SET list (no `+=`, no props folded into
    the MERGE pattern),
  - edge upserts MATCH both endpoints by id with exactly one label each (the
    label the vertex was created with - the live parser requires this) and
    MERGE a directed, single-type relationship carrying
    `{id: row.relationship_vertex}`,
  - node ids are non-negative 63-bit integers, stable across runs,
  - every row in a batch carries exactly the fields the statement reads,
  - property values are limited to int/float/bool/str.

The live round-trip against a real node is tests/test_hydradb_integration.py.
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import networkx as nx
import pytest

from graphify.exporters.hydradb import (
    hydradb_statements,
    hydradb_vertex_id,
    push_to_hydradb,
    write_hydradb_statements,
)

FIXTURES = Path(__file__).parent / "fixtures"

VERTEX_RE = re.compile(
    r"^UNWIND \$rows AS row MERGE \(n \{id: row\.vertex\}\) "
    r"SET n:[A-Za-z_][A-Za-z0-9_]*"
    r"(, n\.[A-Za-z_][A-Za-z0-9_]* = row\.[A-Za-z_][A-Za-z0-9_]*)+$"
)
EDGE_RE = re.compile(
    r"^UNWIND \$rows AS row "
    r"MATCH \(s:[A-Za-z_][A-Za-z0-9_]* \{id: row\.source_vertex\}\), "
    r"\(d:[A-Za-z_][A-Za-z0-9_]* \{id: row\.destination_vertex\}\) "
    r"MERGE \(s\)-\[r:[A-Z_][A-Z0-9_]* \{id: row\.relationship_vertex\}\]->\(d\) "
    r"SET r\.[A-Za-z_][A-Za-z0-9_]* = row\.[A-Za-z_][A-Za-z0-9_]*"
    r"(, r\.[A-Za-z_][A-Za-z0-9_]* = row\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _sample_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("src/a.py", file_type="python", loc=10, ratio=0.5, flagged=True)
    G.add_node("src/b.py", file_type="python", loc=20)
    G.add_node("docs/readme.md", file_type="markdown")
    G.add_edge("src/a.py", "src/b.py", relation="imports", weight=2)
    G.add_edge("src/a.py", "docs/readme.md", relation="mentions")
    G.add_edge("src/b.py", "docs/readme.md", relation="mentions")
    return G


def _communities() -> dict[int, list[str]]:
    return {0: ["src/a.py", "src/b.py"], 1: ["docs/readme.md"]}


def test_statements_stay_inside_compat_subset():
    stmts = hydradb_statements(_sample_graph(), _communities())
    assert stmts, "expected at least one statement"
    for cypher, params in stmts:
        assert VERTEX_RE.match(cypher) or EDGE_RE.match(cypher), cypher
        # Things the subset rejects outright must never appear.
        assert "+=" not in cypher
        assert "OPTIONAL" not in cypher
        assert "--" not in cypher  # undirected patterns are rejected
        assert list(params) == ["rows"]


def test_rows_are_homogeneous_and_scalar():
    for cypher, params in hydradb_statements(_sample_graph(), _communities()):
        rows = params["rows"]
        assert rows
        keysets = {tuple(sorted(r)) for r in rows}
        assert len(keysets) == 1, f"mixed row shapes for: {cypher}"
        for row in rows:
            for value in row.values():
                assert isinstance(value, (int, float, bool, str))


def test_vertex_ids_are_stable_nonnegative_63bit():
    vid = hydradb_vertex_id("src/a.py")
    assert vid == hydradb_vertex_id("src/a.py")
    assert vid != hydradb_vertex_id("src/b.py")
    assert 0 <= vid < 2**63


def test_statements_are_deterministic_across_runs():
    a = hydradb_statements(_sample_graph(), _communities())
    b = hydradb_statements(_sample_graph(), _communities())
    assert a == b


def test_vertices_precede_edges_and_carry_key_and_community():
    stmts = hydradb_statements(_sample_graph(), _communities())
    kinds = ["vertex" if VERTEX_RE.match(c) else "edge" for c, _ in stmts]
    assert kinds == sorted(kinds, key=lambda k: k != "vertex"), (
        "edge statements must come after all vertex statements"
    )
    vertex_rows = [r for c, p in stmts if VERTEX_RE.match(c) for r in p["rows"]]
    by_key = {r["key"]: r for r in vertex_rows}
    assert set(by_key) == {"src/a.py", "src/b.py", "docs/readme.md"}
    assert by_key["src/a.py"]["community"] == 0
    assert by_key["docs/readme.md"]["community"] == 1
    assert by_key["src/a.py"]["vertex"] == hydradb_vertex_id("src/a.py")


def test_reserved_property_names_are_renamed_not_dropped():
    G = nx.DiGraph()
    G.add_node("n1", file_type="python", id="original", vertex="clash")
    stmts = hydradb_statements(G)
    (cypher, params), = stmts
    row = params["rows"][0]
    assert row["p_id"] == "original"
    assert row["p_vertex"] == "clash"
    assert isinstance(row["vertex"], int)
    assert "n.id =" not in cypher  # the identity property is never SET


def test_labels_and_relation_types_are_sanitized():
    G = nx.DiGraph()
    G.add_node("a", file_type="c++ (weird)")
    G.add_node("b", file_type="")
    G.add_edge("a", "b", relation="calls-into thing")
    stmts = hydradb_statements(G)
    cyphers = [c for c, _ in stmts]
    assert any("SET n:C" in c for c in cyphers)
    assert not any(re.search(r":[A-Za-z0-9_]*[^\x00-\x7F]", c) for c in cyphers)
    edge = [c for c in cyphers if EDGE_RE.match(c)]
    assert edge and "-[r:CALLS_INTO_THING {id: row.relationship_vertex}]->" in edge[0]
    # the raw relation text survives as a property value
    edge_rows = [p["rows"][0] for c, p in stmts if EDGE_RE.match(c)]
    assert edge_rows[0]["relation"] == "calls-into thing"


def test_edge_batches_group_by_endpoint_labels():
    """Each edge statement pins one (src label, rel, dst label) combination.

    HydraDB's UNWIND edge batch MATCHes its endpoints with exactly one label
    each, so edges between differently-labeled nodes cannot share a statement.
    """
    stmts = hydradb_statements(_sample_graph(), _communities())
    edge_stmts = [c for c, _ in stmts if EDGE_RE.match(c)]
    assert any("(s:Python {id: row.source_vertex}), (d:Python" in c for c in edge_stmts)
    assert any("(s:Python {id: row.source_vertex}), (d:Markdown" in c for c in edge_stmts)
    # every vertex statement carries exactly one SET label
    for c, _ in stmts:
        if VERTEX_RE.match(c):
            assert len(re.findall(r"SET n:[A-Za-z_]", c)) == 1
            assert ", n:" not in c


def test_parallel_multigraph_edges_get_distinct_relationship_ids():
    """Two same-relation edges between the same nodes (e.g. two call sites)
    must not collide on the id HydraDB MERGEs by. Hashing only
    (src, relation, dst) collapsed them onto one relationship id, and a
    batch that writes that id twice with different properties was rejected
    by a live node with "idempotency key conflict... the batch carries this
    relationship id twice with different endpoints or properties".
    """
    G = nx.MultiDiGraph()
    G.add_node("a", file_type="python")
    G.add_node("b", file_type="python")
    G.add_edge("a", "b", relation="calls", source_location="L433")
    G.add_edge("a", "b", relation="calls", source_location="L459")
    stmts = hydradb_statements(G)
    edge_rows = [r for _, p in stmts for r in p["rows"] if "relationship_vertex" in r]
    assert len(edge_rows) == 2
    assert len({r["relationship_vertex"] for r in edge_rows}) == 2, (
        "parallel edges collapsed onto the same relationship id"
    )
    assert len({r["relationship_id"] for r in edge_rows}) == 2


def test_batch_size_chunks_rows():
    G = nx.DiGraph()
    for i in range(7):
        G.add_node(f"n{i}", file_type="python")
    stmts = hydradb_statements(G, batch_size=3)
    sizes = [len(p["rows"]) for _, p in stmts]
    assert sizes == [3, 3, 1]
    cyphers = {c for c, _ in stmts}
    assert len(cyphers) == 1, "chunks of one group share one statement"


def test_write_hydradb_statements_round_trips(tmp_path):
    path = tmp_path / "hydradb_statements.json"
    n = write_hydradb_statements(_sample_graph(), str(path), _communities())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == n
    assert all(set(item) == {"cypher", "params"} for item in payload)
    expected = hydradb_statements(_sample_graph(), _communities())
    assert [(i["cypher"], i["params"]) for i in payload] == list(expected)


def test_push_to_hydradb_runs_every_statement_over_bolt(monkeypatch):
    """push_to_hydradb sends each compiled statement through one session.run."""
    runs: list[tuple[str, dict]] = []

    class _Result:
        def consume(self):
            return None

    class _Session:
        def run(self, cypher, **params):
            runs.append((cypher, params))
            return _Result()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Driver:
        def __init__(self):
            self.closed = False

        def session(self):
            return _Session()

        def close(self):
            self.closed = True

    driver = _Driver()
    fake = types.ModuleType("neo4j")
    fake.GraphDatabase = types.SimpleNamespace(
        driver=lambda uri, auth: driver if (uri, auth) == (
            "bolt://127.0.0.1:7687", ("neo4j", "token-123")
        ) else pytest.fail(f"unexpected connection args: {uri} {auth}")
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake)

    G = _sample_graph()
    result = push_to_hydradb(
        G, uri="bolt://127.0.0.1:7687", password="token-123",
        communities=_communities(),
    )

    assert result == {"nodes": 3, "edges": 3}
    assert runs == hydradb_statements(G, _communities())
    assert driver.closed


def test_full_fixture_graph_compiles_clean():
    """The real extraction fixture must compile without violating the subset."""
    from graphify.build import build_from_json

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)
    stmts = hydradb_statements(G)
    assert stmts
    seen_vertices = set()
    for cypher, params in stmts:
        assert VERTEX_RE.match(cypher) or EDGE_RE.match(cypher), cypher
        for row in params["rows"]:
            if VERTEX_RE.match(cypher):
                seen_vertices.add(row["vertex"])
    assert len(seen_vertices) == G.number_of_nodes(), "vertex-id collision"
