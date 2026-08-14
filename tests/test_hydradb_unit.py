"""Unit tests for the HydraDB exporter's translation helpers.

These cover the pure parts of push_to_hydradb - id derivation, property
filtering and key sanitization - without needing a running HydraDB node
(tests/test_hydradb_integration.py covers the live path).
"""
from __future__ import annotations

import math

import networkx as nx
import pytest

from graphify.exporters.graphdb import (
    _HYDRADB_ID_MASK,
    _hydradb_id,
    _hydradb_key,
    _hydradb_scalar,
)


def test_hydradb_id_is_deterministic_and_63_bit():
    a = _hydradb_id("app_server")
    assert a == _hydradb_id("app_server")
    assert 0 <= a <= _HYDRADB_ID_MASK
    assert _hydradb_id("app_server") != _hydradb_id("app_server2")
    # Domain separation: an edge key never aliases a same-text node key.
    assert _hydradb_id("edge:a\x00R\x00b") != _hydradb_id("a")


def test_hydradb_id_handles_unicode():
    assert 0 <= _hydradb_id("模块_服务") <= _HYDRADB_ID_MASK


def test_hydradb_scalar_accepts_wire_types():
    assert _hydradb_scalar("s")
    assert _hydradb_scalar("")
    assert _hydradb_scalar(0)
    assert _hydradb_scalar(True)
    assert _hydradb_scalar(1.5)


def test_hydradb_scalar_rejects_unsupported_values():
    assert not _hydradb_scalar(None)
    assert not _hydradb_scalar(math.nan)
    assert not _hydradb_scalar(math.inf)
    assert not _hydradb_scalar([1])
    assert not _hydradb_scalar({"k": 1})
    assert not _hydradb_scalar(1 << 64)  # exceeds Bolt's signed 64-bit range


def test_hydradb_key_sanitizes_to_identifier():
    assert _hydradb_key("source_file") == "source_file"
    assert _hydradb_key("source-file") == "source_file"
    assert _hydradb_key("weird key!") == "weird_key_"
    assert _hydradb_key("0start") == "p_0start"


def test_push_requires_password():
    from graphify.export import push_to_hydradb

    with pytest.raises(ValueError, match="auth token"):
        push_to_hydradb(nx.Graph(), uri="bolt://localhost:7687", password=None)


def test_push_rejects_id_collision(monkeypatch):
    """Two distinct node ids hashing to the same vertex id must be an error,
    not a silent merge of unrelated nodes."""
    pytest.importorskip("neo4j")
    from graphify import exporters

    monkeypatch.setattr(exporters.graphdb, "_hydradb_id", lambda key: 42)
    G = nx.Graph()
    G.add_node("a")
    G.add_node("b")
    with pytest.raises(ValueError, match="collision"):
        exporters.graphdb.push_to_hydradb(
            G, uri="bolt://localhost:7687", password="token"
        )
