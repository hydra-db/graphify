"""graphdb — moved verbatim from graphify/export.py."""
from __future__ import annotations

from graphify.analyze import _node_community_map
import hashlib
import math
import networkx as nx
import re


def push_to_neo4j(
    G: nx.Graph,
    uri: str,
    user: str,
    password: str,
    communities: dict[int, list[str]] | None = None,
) -> dict[str, int]:
    """Push graph directly to a running Neo4j instance via the Python driver.

    Requires: pip install neo4j

    Uses MERGE so re-running is safe - nodes and edges are upserted, not duplicated.
    Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j driver not installed. Run: pip install neo4j"
        ) from e

    node_community = _node_community_map(communities) if communities else {}

    def _safe_rel(relation: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"

    def _safe_label(label: str) -> str:
        """Sanitize a Neo4j node label to prevent Cypher injection."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
        return sanitized if sanitized else "Entity"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes_pushed = 0
    edges_pushed = 0

    with driver.session() as session:
        for node_id, data in G.nodes(data=True):
            props = {
                k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
            }
            props["id"] = node_id
            cid = node_community.get(node_id)
            if cid is not None:
                props["community"] = cid
            ftype = _safe_label(data.get("file_type", "Entity").capitalize())
            session.run(
                f"MERGE (n:{ftype} {{id: $id}}) SET n += $props",
                id=node_id,
                props=props,
            )
            nodes_pushed += 1

        for u, v, data in G.edges(data=True):
            rel = _safe_rel(data.get("relation", "RELATED_TO"))
            props = {
                k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
            }
            session.run(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += $props",
                src=u,
                tgt=v,
                props=props,
            )
            edges_pushed += 1

    driver.close()
    return {"nodes": nodes_pushed, "edges": edges_pushed}

def push_to_falkordb(
    G: nx.Graph,
    uri: str,
    user: str | None = None,
    password: str | None = None,
    communities: dict[int, list[str]] | None = None,
    graph_name: str = "graphify",
) -> dict[str, int]:
    """Push graph directly to a running FalkorDB instance via the Python SDK.

    Requires: pip install falkordb

    FalkorDB is OpenCypher-compatible, so the MERGE/SET upsert queries are
    identical to push_to_neo4j. Differences from the Neo4j path:
      - connects with FalkorDB(host, port, username, password) instead of a bolt
        driver; only the host/port are read from the URI, so the scheme is
        informational - "falkordb://localhost:6379", "redis://localhost:6379"
        and a bare "localhost:6379" are all equivalent (default port 6379).
      - a named graph is selected via db.select_graph(graph_name) (default
        "graphify"); FalkorDB keys each graph by name in the same instance.
      - queries run via graph.query(cypher, params) - there is no session object.
      - auth is optional (FalkorDB runs without credentials by default), so user
        and password may be None.
      - no APOC: the Neo4j path does not use APOC either, so nothing to port.

    Uses MERGE so re-running is safe - nodes and edges are upserted, not
    duplicated. Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from falkordb import FalkorDB
    except ImportError as e:
        raise ImportError(
            "falkordb SDK not installed. Run: pip install falkordb"
        ) from e

    from urllib.parse import urlparse

    node_community = _node_community_map(communities) if communities else {}

    def _safe_rel(relation: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"

    def _safe_label(label: str) -> str:
        """Sanitize a FalkorDB node label to prevent Cypher injection."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
        return sanitized if sanitized else "Entity"

    parsed = urlparse(uri if "://" in uri else f"redis://{uri}")
    # FalkorDB auth is optional. Only send credentials when a password is
    # provided; otherwise connect anonymously and ignore any bolt-style default
    # username (e.g. Neo4j's "neo4j"), which FalkorDB rejects as an unknown ACL
    # user. Credentials embedded in the URI take precedence over the args.
    connect_user = parsed.username or (user if password else None)
    connect_password = parsed.password or (password or None)
    db = FalkorDB(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        username=connect_user,
        password=connect_password,
    )
    graph = db.select_graph(graph_name)
    nodes_pushed = 0
    edges_pushed = 0

    for node_id, data in G.nodes(data=True):
        props = {
            k: v for k, v in data.items()
            if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
        }
        props["id"] = node_id
        cid = node_community.get(node_id)
        if cid is not None:
            props["community"] = cid
        ftype = _safe_label(data.get("file_type", "Entity").capitalize())
        graph.query(
            f"MERGE (n:{ftype} {{id: $id}}) SET n += $props",
            {"id": node_id, "props": props},
        )
        nodes_pushed += 1

    for u, v, data in G.edges(data=True):
        rel = _safe_rel(data.get("relation", "RELATED_TO"))
        props = {
            k: v for k, v in data.items()
            if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
        }
        graph.query(
            f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += $props",
            {"src": u, "tgt": v, "props": props},
        )
        edges_pushed += 1

    return {"nodes": nodes_pushed, "edges": edges_pushed}


# HydraDB ids are non-negative integers on the wire. Bolt PackStream integers
# are signed 64-bit, so ids are folded into [0, 2^63) even though the server
# stores u64 — the top bit is unusable over this transport.
_HYDRADB_ID_MASK = (1 << 63) - 1

# The common label every pushed node carries. HydraDB's UNWIND edge batches
# require exactly one label on each MATCH endpoint, so edges are matched
# through this label rather than the per-node file_type label.
_HYDRADB_BASE_LABEL = "Entity"


def _hydradb_id(key: str) -> int:
    """Map an arbitrary graphify node/edge key to a stable HydraDB id."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _HYDRADB_ID_MASK


def _hydradb_scalar(value) -> bool:
    """True when the value survives HydraDB's property model.

    HydraDB properties are integers, finite floats, booleans and strings;
    null is rejected outright, and non-finite floats fail parameter
    validation on the server.
    """
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return -(1 << 63) <= value < (1 << 63)
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, str)


def _hydradb_key(key: str) -> str | None:
    """Sanitize a property key so it can appear in `SET n.<key> = row.<key>`.

    UNWIND batch statements embed the key in the query text (there is no
    `SET n += $props` in HydraDB), so it must be a plain identifier.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", key)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"p_{sanitized}"
    return sanitized


def push_to_hydradb(
    G: nx.Graph,
    uri: str,
    user: str = "neo4j",
    password: str | None = None,
    communities: dict[int, list[str]] | None = None,
    database: str = "default",
    batch_size: int = 500,
) -> dict[str, int]:
    """Push graph to a running HydraDB node over Bolt.

    Requires: pip install neo4j  (HydraDB speaks Bolt 5.1-5.4, so the Neo4j
    driver is the client; `password` is HydraDB's shared auth token and `user`
    may be any non-empty string).

    HydraDB executes a deliberate subset of OpenCypher (see cypher-compat.md in
    the HydraDB repo), so this is a translation layer rather than a copy of the
    Neo4j path:

      - node ids must be non-negative integers, so every graphify node id is
        hashed to a stable 63-bit integer and the original string id is kept
        as the `uid` property. Edge ids are hashed the same way from
        (source, relation, target), which keeps re-pushes idempotent.
      - there is no `SET n += $props`; writes go through HydraDB's batched
        `UNWIND $rows` upsert forms, which read every value from the row map.
        Rows are grouped by their exact property-key set so each statement
        can name each key.
      - a vertex upsert sets exactly one label per statement. Every node gets
        the shared `Entity` label (edge endpoint MATCHes require a label) and
        a second label-only pass adds the per-node file_type label.
      - properties are limited to integers, finite floats, booleans and
        strings; null and non-finite values are dropped like the other
        exporters drop non-scalars.

    Uses MERGE so re-running is safe - nodes and edges are upserted, not
    duplicated. Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j driver not installed (HydraDB uses the Bolt protocol). "
            "Run: pip install neo4j"
        ) from e
    if not password:
        raise ValueError(
            "HydraDB requires its auth token as the password "
            "(the server rejects unauthenticated Bolt sessions)"
        )

    node_community = _node_community_map(communities) if communities else {}

    def _safe_rel(relation: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"

    def _safe_label(label: str) -> str:
        """Sanitize a HydraDB node label to prevent Cypher injection."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
        return sanitized if sanitized else "Entity"

    # ------- plan node rows -------
    # uid -> integer id, with an explicit collision check: HydraDB would
    # otherwise silently merge two distinct nodes into one vertex.
    vertex_ids: dict[str, int] = {}
    claimed: dict[int, str] = {}
    for node_id in G.nodes:
        uid = str(node_id)
        vid = _hydradb_id(uid)
        other = claimed.get(vid)
        if other is not None and other != uid:
            raise ValueError(
                f"HydraDB id collision between nodes {other!r} and {uid!r}; "
                "rename one of them and re-export"
            )
        claimed[vid] = uid
        vertex_ids[uid] = vid

    # Group rows by their exact property-key set: every row in an UNWIND batch
    # must carry every field the statement reads.
    node_groups: dict[tuple[str, ...], list[dict]] = {}
    type_labels: dict[str, list[int]] = {}
    for node_id, data in G.nodes(data=True):
        uid = str(node_id)
        props = {}
        for k, v in data.items():
            if k.startswith("_") or not _hydradb_scalar(v):
                continue
            sk = _hydradb_key(k)
            if sk is not None and sk not in ("id", "vertex"):
                props[sk] = v
        props["uid"] = uid
        cid = node_community.get(node_id)
        if cid is not None:
            props["community"] = cid
        ftype = _safe_label(str(data.get("file_type", "Entity")).capitalize())
        if ftype != _HYDRADB_BASE_LABEL:
            type_labels.setdefault(ftype, []).append(vertex_ids[uid])
        row = {"vertex": vertex_ids[uid], **props}
        node_groups.setdefault(tuple(sorted(props)), []).append(row)

    # ------- plan edge rows -------
    edge_claimed: dict[int, tuple[str, str, str]] = {}
    edge_groups: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for u, v, data in G.edges(data=True):
        rel = _safe_rel(data.get("relation", "RELATED_TO"))
        triple = (str(u), rel, str(v))
        eid = _hydradb_id("edge:{}\x00{}\x00{}".format(*triple))
        other_triple = edge_claimed.get(eid)
        if other_triple is not None and other_triple != triple:
            raise ValueError(
                f"HydraDB id collision between edges {other_triple!r} and {triple!r}"
            )
        edge_claimed[eid] = triple
        props = {}
        for k, val in data.items():
            if k.startswith("_") or not _hydradb_scalar(val):
                continue
            sk = _hydradb_key(k)
            if sk is not None and sk not in ("id", "eid", "src", "dst"):
                props[sk] = val
        row = {
            "src": vertex_ids[str(u)],
            "dst": vertex_ids[str(v)],
            "eid": eid,
            **props,
        }
        edge_groups.setdefault((rel, tuple(sorted(props))), []).append(row)

    # ------- execute -------
    driver = GraphDatabase.driver(uri, auth=(user or "neo4j", password))
    nodes_pushed = 0
    edges_pushed = 0
    try:
        with driver.session(database=database) as session:
            for keys, rows in node_groups.items():
                set_items = ", ".join(f"n.{k} = row.{k}" for k in keys)
                query = (
                    "UNWIND $rows AS row MERGE (n {id: row.vertex}) "
                    f"SET n:{_HYDRADB_BASE_LABEL}, {set_items}"
                )
                for start in range(0, len(rows), batch_size):
                    chunk = rows[start:start + batch_size]
                    session.run(query, rows=chunk).consume()
                    nodes_pushed += len(chunk)

            for label, vids in type_labels.items():
                query = (
                    "UNWIND $rows AS row MERGE (n {id: row.vertex}) "
                    f"SET n:{label}"
                )
                for start in range(0, len(vids), batch_size):
                    chunk = [{"vertex": vid} for vid in vids[start:start + batch_size]]
                    session.run(query, rows=chunk).consume()

            for (rel, keys), rows in edge_groups.items():
                set_clause = ""
                if keys:
                    set_items = ", ".join(f"r.{k} = row.{k}" for k in keys)
                    set_clause = f" SET {set_items}"
                query = (
                    "UNWIND $rows AS row "
                    f"MATCH (s:{_HYDRADB_BASE_LABEL} {{id: row.src}}), "
                    f"(d:{_HYDRADB_BASE_LABEL} {{id: row.dst}}) "
                    f"MERGE (s)-[r:{rel} {{id: row.eid}}]->(d){set_clause}"
                )
                for start in range(0, len(rows), batch_size):
                    chunk = rows[start:start + batch_size]
                    session.run(query, rows=chunk).consume()
                    edges_pushed += len(chunk)
    finally:
        driver.close()
    return {"nodes": nodes_pushed, "edges": edges_pushed}
