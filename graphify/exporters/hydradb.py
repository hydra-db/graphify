"""HydraDB (core, open-source) graph sink.

HydraDB is an object-store-native graph database that speaks Bolt 5.x, so the
Neo4j Python driver connects to it unchanged. What *does* change is the query
surface: HydraDB implements a deliberate subset of OpenCypher (see
`cypher-compat.md` in the hydra-db/hydradb repository), and the generic
Neo4j/FalkorDB upsert queries used elsewhere in this package are rejected by
its parser. The constraints that shape this module:

  - Node ids are non-negative integers, and a pattern matches on `id`.
    Graphify node ids are strings (file paths / symbols), so each node gets a
    stable 63-bit hash id and keeps its original string id in the `key`
    property.
  - `SET n += $props` does not exist; every property is a separate
    `n.prop = row.prop` assignment, so each statement fixes its property set
    and rows must be grouped by exact property-key set.
  - Bulk writes go through `UNWIND $rows AS row ...` with the list-of-maps
    parameter, which HydraDB only accepts over the client (Bolt) transport:
      vertices:  UNWIND $rows AS row MERGE (n {id: row.vertex})
                 SET n:Label, n.key = row.key, ...
      edges:     UNWIND $rows AS row
                 MATCH (s:SrcLabel {id: row.source_vertex}), (d:DstLabel {id: row.destination_vertex})
                 MERGE (s)-[r:REL {id: row.relationship_vertex}]->(d)
                 SET r.relation = row.relation, ...
  - A vertex upsert must be MERGE-by-id followed by SET; folding other
    properties into the MERGE pattern is rejected. The live parser enforces
    exactly one SET label per vertex upsert, and the edge batch's MATCH
    endpoints require exactly one label each - the label the vertex was
    created with - so edge batches are additionally grouped by their
    endpoint labels.
  - Relationship patterns carry exactly one type and a direction, and only one
    statement is accepted per request.

Auth: a single bearer token (GRAPH_AUTH_TOKEN_FILE on the server). Bolt
clients pass it as the password with user "neo4j", e.g.
`neo4j.GraphDatabase.driver(uri, auth=("neo4j", token))`.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import networkx as nx

from graphify.analyze import _node_community_map

# Row fields the generated statements read for identity; data properties are
# renamed away from these so a node attribute can never shadow the id wiring.
_RESERVED_ROW_FIELDS = frozenset(
    {"vertex", "source_vertex", "destination_vertex", "relationship_vertex"}
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Default rows per UNWIND statement. HydraDB commits each statement durably to
# object storage, so fewer round trips matter more than statement size.
DEFAULT_BATCH_SIZE = 1000


def _safe_rel(relation: str) -> str:
    return re.sub(
        r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")
    ) or "RELATED_TO"


def _safe_label(label: str) -> str:
    """Sanitize a node label to prevent Cypher injection."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
    return sanitized if sanitized else "Entity"


def _safe_prop(name: str) -> str | None:
    """Return a statement-safe property name, or None to drop the property.

    Property names are interpolated into the statement (`SET n.<name> =
    row.<name>`), so they must be plain identifiers. Names colliding with the
    reserved row fields are prefixed rather than dropped so the data survives.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"p_{sanitized}"
    if not _IDENT_RE.match(sanitized):
        return None
    if sanitized in _RESERVED_ROW_FIELDS or sanitized == "id":
        # `id` is the integer identity HydraDB matched on and cannot be SET.
        sanitized = f"p_{sanitized}"
    return sanitized


def hydradb_vertex_id(key: str) -> int:
    """Stable non-negative 63-bit integer id for a graphify node id.

    HydraDB node ids are non-negative integers. Hashing (rather than
    enumerating) keeps ids stable when the graph is rebuilt with nodes added
    or removed, which is what makes MERGE re-runs converge on the same
    vertices.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def _edge_id(src_key: str, relation: str, dst_key: str, edge_key) -> int:
    """Hash the full multigraph identity, including the parallel-edge key.

    graphify builds a MultiDiGraph: two symbols can be connected by more than
    one edge of the same relation (e.g. `a` calls `b` from two call sites).
    Hashing only (src, relation, dst) collapsed those onto the same HydraDB
    relationship id, and a batch that writes the same id twice with different
    properties is rejected outright ("idempotency key conflict"). `edge_key`
    is networkx's own per-(src, dst) disambiguator, so folding it in makes
    every parallel edge distinct without changing the id for a true re-push
    of the same edge.
    """
    payload = f"{src_key}\x00{relation}\x00{dst_key}\x00{edge_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"edge").digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def _relationship_id(src_key: str, relation: str, dst_key: str, edge_key) -> str:
    """Human-inspectable identity for the `relationship_id` property.

    HydraDB's own batch examples (cypher-compat.md) carry a `relationship_id`
    property alongside the opaque `id` used for MERGE - the id is what the
    pattern matches on, this is what a person or log line can read. Same
    inputs as `_edge_id`, just not hashed.
    """
    return f"{src_key}|{relation}|{dst_key}|{edge_key}"


def _scalar_props(data: dict) -> dict:
    """Filter node/edge attributes to HydraDB's value types, with safe names."""
    props: dict = {}
    for k, v in data.items():
        if k.startswith("_") or not isinstance(v, (str, int, float, bool)):
            continue
        safe = _safe_prop(str(k))
        if safe is None or safe in props:
            continue
        props[safe] = v
    return props


def _chunks(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def hydradb_statements(
    G: nx.Graph,
    communities: dict[int, list[str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[tuple[str, dict]]:
    """Compile a graph into HydraDB-compatible `(cypher, params)` statements.

    Returns the full ordered batch plan: vertex upserts first (grouped by
    label and exact property-key set, as required by the fixed SET list), then
    edge upserts (grouped by relationship type and key set). Deterministic for
    a given graph, so re-running produces the same statements and MERGE makes
    the push idempotent.
    """
    node_community = _node_community_map(communities) if communities else {}

    # ---- vertices, grouped by (label, property key set) ----
    vertex_groups: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    node_labels: dict[str, str] = {}
    for node_id, data in G.nodes(data=True):
        props = _scalar_props(data)
        # The original string id always travels along; `id` is the hash. A
        # data attribute literally named "id" is renamed to "p_id" upstream in
        # _safe_prop, because `n.id` is the identity and cannot be SET.
        props["key"] = str(node_id)
        cid = node_community.get(node_id)
        if cid is not None:
            props["community"] = int(cid)
        label = _safe_label(str(data.get("file_type", "Entity")).capitalize())
        node_labels[str(node_id)] = label
        keyset = tuple(sorted(props))
        row = {"vertex": hydradb_vertex_id(str(node_id))}
        row.update(props)
        vertex_groups.setdefault((label, keyset), []).append(row)

    statements: list[tuple[str, dict]] = []
    for (label, keyset) in sorted(vertex_groups):
        rows = vertex_groups[(label, keyset)]
        rows.sort(key=lambda r: r["vertex"])
        sets = ", ".join(f"n.{k} = row.{k}" for k in keyset)
        cypher = (
            f"UNWIND $rows AS row MERGE (n {{id: row.vertex}}) "
            f"SET n:{label}, {sets}"
        )
        for chunk in _chunks(rows, batch_size):
            statements.append((cypher, {"rows": chunk}))

    # ---- edges, grouped by (endpoint labels, relationship type, key set) ----
    # The MATCH endpoints of an UNWIND edge batch require exactly one label
    # each, and it must be the label the vertex carries, so the endpoint
    # labels are part of the statement and therefore of the grouping key.
    edge_groups: dict[tuple[str, str, str, tuple[str, ...]], list[dict]] = {}
    # G is a MultiDiGraph - two nodes can carry more than one edge of the same
    # relation, disambiguated by networkx's own per-(src, dst) edge key. A
    # plain (non-multi) graph has no such key, so it gets a fixed stand-in.
    if G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in G.edges(data=True))
    for u, v, ekey, data in edge_iter:
        relation = str(data.get("relation", "RELATED_TO"))
        rel = _safe_rel(relation)
        props = _scalar_props(data)
        # Always at least one SET column, and the pre-sanitization relation
        # text is worth keeping anyway.
        props["relation"] = relation
        props["relationship_id"] = _relationship_id(str(u), relation, str(v), ekey)
        keyset = tuple(sorted(props))
        row = {
            "source_vertex": hydradb_vertex_id(str(u)),
            "destination_vertex": hydradb_vertex_id(str(v)),
            "relationship_vertex": _edge_id(str(u), relation, str(v), ekey),
        }
        row.update(props)
        src_label = node_labels.get(str(u), "Entity")
        dst_label = node_labels.get(str(v), "Entity")
        edge_groups.setdefault((src_label, rel, dst_label, keyset), []).append(row)

    for (src_label, rel, dst_label, keyset) in sorted(edge_groups):
        rows = edge_groups[(src_label, rel, dst_label, keyset)]
        rows.sort(key=lambda r: r["relationship_vertex"])
        sets = ", ".join(f"r.{k} = row.{k}" for k in keyset)
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (s:{src_label} {{id: row.source_vertex}}), "
            f"(d:{dst_label} {{id: row.destination_vertex}}) "
            f"MERGE (s)-[r:{rel} {{id: row.relationship_vertex}}]->(d) "
            f"SET {sets}"
        )
        for chunk in _chunks(rows, batch_size):
            statements.append((cypher, {"rows": chunk}))

    return statements


def write_hydradb_statements(
    G: nx.Graph,
    path: str,
    communities: dict[int, list[str]] | None = None,
) -> int:
    """Write the batch plan as JSON for replay through any Bolt client.

    HydraDB accepts one statement per request and the UNWIND list-of-maps
    parameter only over the client transport, so a flat cypher script (like
    `cypher.txt` for Neo4j) cannot express this plan. The JSON file carries
    `{"cypher": ..., "params": ...}` objects instead.
    """
    statements = hydradb_statements(G, communities)
    payload = [{"cypher": c, "params": p} for c, p in statements]
    Path(path).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return len(statements)


def push_to_hydradb(
    G: nx.Graph,
    uri: str,
    user: str = "neo4j",
    password: str | None = None,
    communities: dict[int, list[str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """Push graph to a running HydraDB node over Bolt via the Neo4j driver.

    Requires: pip install neo4j

    `password` is the HydraDB auth token (the content of the server's
    GRAPH_AUTH_TOKEN_FILE); `user` is ignored by HydraDB but the Bolt
    handshake needs one, and "neo4j" is what its own smoke tests use. A local
    development node listens on bolt://127.0.0.1:7687 with
    GRAPH_ALLOW_PLAINTEXT=true.

    Uses MERGE-by-id batches so re-running is safe - nodes and edges are
    upserted, not duplicated. Returns a dict with counts of nodes and edges
    pushed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j driver not installed (HydraDB speaks Bolt). Run: pip install neo4j"
        ) from e

    statements = hydradb_statements(G, communities, batch_size=batch_size)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for cypher, params in statements:
                session.run(cypher, **params).consume()
    finally:
        driver.close()
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
