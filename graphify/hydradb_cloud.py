"""HydraDB managed-platform integration (api.hydradb.com).

HydraDB's managed offering is a context platform for AI agents: you provision
an isolated *database*, ingest *knowledge* (documents) or *memories*, and
retrieve ranked chunks - optionally with graph context (entity triplets and
relation paths) - through a single /query endpoint. This module gives
graphify a first-class sink and retrieval surface for it:

  - ``HydraDBCloudClient``: a stdlib-only v2 REST client (Authorization:
    Bearer + ``API-Version: 2``), unwrapping the ``{success, data, error,
    meta}`` envelope and retrying only 429/500/503 with exponential backoff,
    as the platform's integration guide specifies.
  - ``sync_out_dir``: pushes a graphify output directory (GRAPH_REPORT.md and
    the agent wiki) into a HydraDB database as knowledge documents, so any
    agent wired to HydraDB can retrieve the graph's insights alongside its
    other context.

The API key comes from the ``HYDRA_DB_API_KEY`` environment variable (the
name the platform documents) and is never written to disk or argv.

Wire notes (from the platform's OpenAPI spec, API-Version 2):
  - ``database`` is the canonical workspace field (``tenant_id`` is a
    deprecated alias); ``collection`` partitions within a database.
  - ``POST /databases`` is async: poll ``GET /databases/status`` until
    ``infra.ready_for_ingestion`` before ingesting.
  - ``POST /context/ingest`` is multipart; files ride the repeated
    ``documents`` field and indexing completes asynchronously - poll
    ``GET /context/status`` for each returned source id.
  - ``POST /query`` selects sources via ``type`` (knowledge | memory | all),
    matches via ``query_by`` (hybrid combines dense + BM25), and returns
    graph context by default.
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE_URL = "https://api.hydradb.com"
API_KEY_ENV = "HYDRA_DB_API_KEY"
# Only these statuses are retried; everything else is a real answer from the
# platform (auth, validation, not-found) that retrying cannot fix.
_RETRYABLE_STATUS = frozenset({429, 500, 503})


class HydraDBCloudError(RuntimeError):
    """A failed HydraDB API call, carrying the platform's error envelope."""

    def __init__(self, message: str, status: int | None = None,
                 code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


def _encode_multipart(fields: dict[str, str],
                      files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """Encode a multipart/form-data body with the stdlib.

    ``files`` entries are ``(field_name, filename, content)``. Returns
    ``(body, content_type)``.
    """
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        buf.write(str(value).encode("utf-8"))
        buf.write(b"\r\n")
    for name, filename, content in files:
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'.encode()
        )
        buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        buf.write(content)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


class HydraDBCloudClient:
    """Minimal typed client for the HydraDB managed platform (API v2)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff: float = 1.0,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise HydraDBCloudError(
                f"no API key: pass api_key or set {API_KEY_ENV}"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        # Response meta of the most recent successful call (request_id etc.).
        self.last_meta: dict = {}

    # -- transport -----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        params: dict | None = None,
        multipart: tuple[dict, list] | None = None,
    ) -> dict:
        """One API call; returns the envelope's ``data``.

        Retries 429/500/503 with exponential backoff (honouring Retry-After
        when sent); raises :class:`HydraDBCloudError` for everything else and
        for ``success: false`` envelopes.
        """
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "API-Version": "2",
            "Accept": "application/json",
        }
        body: bytes | None = None
        if multipart is not None:
            fields, files = multipart
            body, ctype = _encode_multipart(fields, files)
            headers["Content-Type"] = ctype
        elif json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return self._unwrap(resp.read(), resp.status)
            except urllib.error.HTTPError as e:
                status = e.code
                payload = e.read()
                if status in _RETRYABLE_STATUS and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self.backoff * (2 ** attempt)
                    )
                    time.sleep(delay)
                    last_err = e
                    continue
                raise self._envelope_error(payload, status) from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    last_err = e
                    continue
                raise HydraDBCloudError(f"connection failed: {e}") from e
        raise HydraDBCloudError(f"retries exhausted: {last_err}")

    @staticmethod
    def _envelope_error(payload: bytes, status: int) -> HydraDBCloudError:
        try:
            envelope = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return HydraDBCloudError(
                f"HTTP {status}: {payload[:200]!r}", status=status
            )
        err = envelope.get("error") or {}
        detail = envelope.get("detail") or {}
        message = (
            err.get("message") or detail.get("message")
            or envelope.get("message") or f"HTTP {status}"
        )
        code = err.get("code") or detail.get("error_code")
        return HydraDBCloudError(message, status=status, code=code)

    def _unwrap(self, payload: bytes, status: int) -> dict:
        envelope = json.loads(payload)
        if not envelope.get("success", True):
            raise self._envelope_error(payload, status)
        # Keep the response meta (request_id, latency) reachable: /feedback
        # requires the request_id that /query's meta carried.
        self.last_meta = envelope.get("meta") or {}
        return envelope.get("data") or {}

    # -- databases -----------------------------------------------------------

    def list_databases(self) -> list[str]:
        data = self._request("GET", "/databases")
        return data.get("databases") or data.get("tenant_ids") or []

    def create_database(self, database: str,
                        metadata_schema: list[dict] | None = None) -> dict:
        body: dict = {"database": database}
        if metadata_schema:
            body["database_metadata_schema"] = metadata_schema
        return self._request("POST", "/databases", json_body=body)

    def database_status(self, database: str) -> dict:
        return self._request("GET", "/databases/status",
                             params={"database": database})

    def delete_database(self, database: str) -> dict:
        return self._request("DELETE", "/databases",
                             params={"database": database})

    def wait_until_ready(self, database: str, timeout: float = 300.0,
                         interval: float = 5.0) -> dict:
        """Poll /databases/status until infra accepts ingestion."""
        deadline = time.monotonic() + timeout
        while True:
            status = self.database_status(database)
            infra = status.get("infra") or {}
            if infra.get("ready_for_ingestion"):
                return status
            if time.monotonic() >= deadline:
                raise HydraDBCloudError(
                    f"database {database!r} not ready after {timeout:.0f}s: "
                    f"{json.dumps(infra)}"
                )
            time.sleep(interval)

    # -- ingestion -----------------------------------------------------------

    def ingest_files(
        self,
        database: str,
        files: list[Path],
        collection: str | None = None,
        document_metadata: list[dict] | None = None,
        upsert: bool = True,
    ) -> dict:
        """Upload documents as knowledge; returns per-file results with ids."""
        fields = {"database": database, "type": "knowledge",
                  "upsert": "true" if upsert else "false"}
        if collection:
            fields["collection"] = collection
        if document_metadata:
            fields["document_metadata"] = json.dumps(document_metadata)
        parts = [
            ("documents", Path(f).name, Path(f).read_bytes()) for f in files
        ]
        return self._request("POST", "/context/ingest",
                             multipart=(fields, parts))

    def ingest_memories(self, database: str, memories: list[dict],
                        collection: str | None = None) -> dict:
        fields = {"database": database, "type": "memory",
                  "memories": json.dumps(memories)}
        if collection:
            fields["collection"] = collection
        return self._request("POST", "/context/ingest",
                             multipart=(fields, []))

    def context_status(self, database: str, ids: list[str] | None = None,
                       collection: str | None = None) -> list[dict]:
        params: dict = {"database": database}
        if collection:
            params["collection"] = collection
        if ids:
            params["ids"] = ",".join(ids)
        data = self._request("GET", "/context/status", params=params)
        return data.get("statuses") or []

    def wait_for_indexing(self, database: str, ids: list[str],
                          timeout: float = 600.0,
                          interval: float = 5.0) -> list[dict]:
        """Poll /context/status until every source finishes (or fails)."""
        deadline = time.monotonic() + timeout
        while True:
            statuses = self.context_status(database, ids=ids)
            by_id = {s.get("id"): s for s in statuses}
            pending = [
                i for i in ids
                if (by_id.get(i) or {}).get("indexing_status")
                not in ("completed", "failed")
            ]
            if not pending:
                return [by_id[i] for i in ids if i in by_id]
            if time.monotonic() >= deadline:
                raise HydraDBCloudError(
                    f"indexing incomplete after {timeout:.0f}s; "
                    f"still pending: {pending}"
                )
            time.sleep(interval)

    def list_context(self, database: str, collection: str | None = None,
                     page: int = 1, page_size: int = 50) -> dict:
        body: dict = {"database": database, "page": page,
                      "page_size": page_size}
        if collection:
            body["collection"] = collection
        return self._request("POST", "/context/list", json_body=body)

    def delete_context(self, database: str, ids: list[str],
                       collection: str | None = None) -> dict:
        body: dict = {"database": database, "ids": ids}
        if collection:
            body["collection"] = collection
        return self._request("DELETE", "/context", json_body=body)

    # -- retrieval -----------------------------------------------------------

    def query(
        self,
        database: str,
        query: str,
        type: str = "knowledge",
        query_by: str = "hybrid",
        mode: str | None = None,
        max_results: int | None = None,
        collection: str | None = None,
        graph_context: bool = True,
        additional_context: str | None = None,
        recency_bias: float | None = None,
    ) -> dict:
        """Retrieve ranked chunks (and graph context) for a question.

        ``type`` selects knowledge, memory, or all; ``query_by: "hybrid"``
        fuses dense and BM25 retrieval; ``mode: "thinking"`` adds query
        expansion and graph traversal at the cost of latency.
        """
        body: dict = {
            "database": database,
            "query": query,
            "type": type,
            "query_by": query_by,
            "graph_context": graph_context,
        }
        if mode:
            body["mode"] = mode
        if max_results is not None:
            body["max_results"] = max_results
        if collection:
            body["collection"] = collection
        if additional_context:
            body["additional_context"] = additional_context
        if recency_bias is not None:
            body["recency_bias"] = recency_bias
        data = self._request("POST", "/query", json_body=body)
        # Surface the id /feedback needs; the data payload never carries one.
        data.setdefault("request_id", self.last_meta.get("request_id"))
        return data

    def feedback(
        self,
        database: str,
        request_id: str,
        feedback: str | None = None,
        ground_truth: dict | None = None,
        rating: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Report retrieval quality for one query.

        ``request_id`` ties the signal to a specific /query call - it is the
        ``request_id`` field :meth:`query` returns (from the response meta),
        and the platform rejects feedback without it. Needs free text or
        ground truth.
        """
        if not request_id:
            raise HydraDBCloudError(
                "feedback needs the request_id returned by query()"
            )
        if not feedback and not ground_truth:
            raise HydraDBCloudError(
                "feedback needs a free-text comment or ground_truth"
            )
        body: dict = {"database": database, "request_id": request_id}
        if feedback:
            body["feedback"] = feedback
        if ground_truth:
            body["ground_truth"] = ground_truth
        if rating:
            body["rating"] = rating
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/feedback", json_body=body)


# -- graphify-specific sync ---------------------------------------------------

def default_database_name(project_dir: str | Path) -> str:
    """Stable HydraDB database name for a project directory."""
    stem = Path(project_dir).resolve().name.lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-_") or "project"
    return f"graphify-{slug}"


def collect_sync_files(out_dir: str | Path) -> list[Path]:
    """Graph artifacts worth retrieving as knowledge: report + agent wiki.

    graph.html/graph.json stay local - the platform indexes prose, and the
    wiki articles plus GRAPH_REPORT.md are the prose form of the graph.
    """
    out = Path(out_dir)
    files: list[Path] = []
    report = out / "GRAPH_REPORT.md"
    if report.exists():
        files.append(report)
    wiki = out / "wiki"
    if wiki.is_dir():
        files.extend(sorted(wiki.glob("*.md")))
    return files


def sync_out_dir(
    client: HydraDBCloudClient,
    out_dir: str | Path,
    database: str,
    collection: str | None = None,
    wait: bool = True,
    log=print,
) -> dict:
    """Push a graphify output directory into a HydraDB database.

    Provisions the database when absent, waits until infra accepts data,
    uploads GRAPH_REPORT.md and the wiki as knowledge documents (upsert, so
    re-syncing after a rebuild updates in place), and optionally waits for
    indexing to complete. Returns a summary dict.
    """
    files = collect_sync_files(out_dir)
    if not files:
        raise HydraDBCloudError(
            f"nothing to sync in {out_dir}: expected GRAPH_REPORT.md and/or "
            f"wiki/*.md (run /graphify with --wiki first)"
        )

    if database not in client.list_databases():
        log(f"creating HydraDB database {database!r} ...")
        client.create_database(database)
    client.wait_until_ready(database)

    log(f"ingesting {len(files)} documents into {database!r} ...")
    result = client.ingest_files(database, files, collection=collection)
    results = result.get("results") or []
    ids = [r["id"] for r in results if r.get("id")]
    failed = [r for r in results if r.get("error")]

    statuses: list[dict] = []
    if wait and ids:
        log(f"waiting for indexing of {len(ids)} sources ...")
        statuses = client.wait_for_indexing(database, ids)
        failed.extend(
            s for s in statuses if s.get("indexing_status") == "failed"
        )

    return {
        "database": database,
        "files": [str(f) for f in files],
        "ids": ids,
        "failed": failed,
        "statuses": statuses,
    }


def format_query_result(data: dict, verbose: bool = False) -> str:
    """Human-readable rendering of a /query response for the CLI."""
    lines: list[str] = []
    chunks = data.get("chunks") or []
    if not chunks:
        lines.append("no results")
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("source_title") or chunk.get("id") or "?"
        score = chunk.get("relevancy_score")
        score_s = f" (score {score:.2f})" if isinstance(score, (int, float)) else ""
        lines.append(f"[{i}] {title}{score_s}")
        content = (chunk.get("chunk_content") or "").strip()
        if not verbose and len(content) > 500:
            content = content[:500] + " ..."
        lines.extend(f"    {line}" for line in content.splitlines())
        lines.append("")
    graph = data.get("graph_context") or {}
    triplets = [
        t
        for rel in (graph.get("chunk_relations") or [])
        for t in (rel.get("triplets") or [])
    ]
    if triplets:
        lines.append("graph context:")
        for t in triplets[:20]:
            src = (t.get("source") or {}).get("name", "?")
            dst = (t.get("target") or {}).get("name", "?")
            rel = t.get("relation") or {}
            pred = (
                rel.get("canonical_predicate")
                or rel.get("raw_predicate")
                or rel.get("predicate")
                or "?"
            )
            lines.append(f"  {src} --{pred}--> {dst}")
    if data.get("request_id"):
        lines.append("")
        lines.append(f"request id: {data['request_id']} (for feedback)")
    return "\n".join(lines).rstrip()
