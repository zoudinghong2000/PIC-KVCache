"""Single-host lookup RPC used between the scheduler and TP workers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from .storage import LocalPinnedCPUStore
from .types import BlendSegment

logger = logging.getLogger(__name__)


def endpoint_path(root: str, engine_id: str, rank: int) -> Path:
    safe_engine = "".join(
        character for character in engine_id if character.isalnum() or character in "-_"
    )
    if not safe_engine:
        raise ValueError("engine_id contains no path-safe characters")
    return Path(root) / safe_engine[:48] / f"rank-{rank}.sock"


def endpoint_uri(root: str, engine_id: str, rank: int) -> str:
    return f"ipc://{endpoint_path(root, engine_id, rank)}"


class LookupServer:
    def __init__(
        self,
        root: str,
        engine_id: str,
        rank: int,
        store: LocalPinnedCPUStore,
        uri: str | None = None,
    ):
        self.path = endpoint_path(root, engine_id, rank) if uri is None else None
        self.uri = endpoint_uri(root, engine_id, rank) if uri is None else uri
        self.store = store
        self._context = zmq.Context.instance()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name=f"cacheblend-lookup-rank-{rank}",
            daemon=True,
        )

    def start(self, timeout: float = 10.0) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.parent.chmod(0o700)
            self.path.unlink(missing_ok=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError(f"CacheBlend lookup server did not bind {self.uri}")
        if self._error is not None:
            raise RuntimeError(
                f"CacheBlend lookup server failed to bind {self.uri}"
            ) from self._error
        if self.path is not None:
            self.path.chmod(0o600)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            client = LookupClient(self.uri, timeout_ms=500)
            try:
                client.request({"op": "SHUTDOWN"})
            except (RuntimeError, TypeError, zmq.ZMQError):
                logger.debug("Lookup server %s was already closed", self.uri)
            finally:
                client.close()
            self._thread.join(timeout=2)
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def _serve(self) -> None:
        socket = self._context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            try:
                socket.bind(self.uri)
            except zmq.ZMQError as error:
                self._error = error
                return
            self._ready.set()
            while not self._stop.is_set():
                try:
                    message = socket.recv_pyobj()
                    response = self._handle(message)
                except Exception as error:
                    logger.exception("CacheBlend lookup request failed")
                    response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                socket.send_pyobj(response)
        finally:
            socket.close(0)
            self._ready.set()

    def _handle(self, message: dict[str, Any]) -> dict[str, Any]:
        operation = message["op"]
        if operation == "SHUTDOWN":
            self._stop.set()
            return {"ok": True}
        if operation == "MATCH":
            raw_tokens = message["tokens"]
            tokens = (
                raw_tokens
                if isinstance(raw_tokens, np.ndarray)
                else np.asarray(raw_tokens, dtype=np.int64)
            )
            segments = self.store.match(
                str(message["model_scope"]),
                tokens,
                int(message["start_offset"]),
            )
            return {"ok": True, "segments": [segment.to_dict() for segment in segments]}
        segments = tuple(BlendSegment.from_dict(value) for value in message.get("segments", []))
        if operation == "VALIDATE":
            valid = self.store.validate(segments)
            return {"ok": True, "segments": [segment.to_dict() for segment in valid]}
        if operation == "PREFETCH":
            valid = self.store.pin(str(message["request_id"]), segments)
            return {"ok": True, "segments": [segment.to_dict() for segment in valid]}
        if operation == "CANCEL":
            self.store.release(str(message["request_id"]))
            return {"ok": True}
        raise ValueError(f"unknown CacheBlend IPC operation {operation!r}")


class LookupClient:
    """Persistent REQ client with LMCache-style recovery after an RPC error."""

    def __init__(self, uri: str, timeout_ms: int = 10_000):
        self.uri = uri
        self.timeout_ms = timeout_ms
        self._context = zmq.Context.instance()
        self._socket = self._create_socket()

    def _create_socket(self) -> zmq.Socket:
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.uri)
        return socket

    def reset(self) -> None:
        self._socket.close(0)
        self._socket = self._create_socket()

    def close(self) -> None:
        self._socket.close(0)

    def send(self, message: dict[str, Any]) -> None:
        try:
            self._socket.send_pyobj(message)
        except zmq.ZMQError:
            self.reset()
            raise

    def receive(self) -> dict[str, Any]:
        try:
            response = self._socket.recv_pyobj()
        except zmq.ZMQError:
            self.reset()
            raise
        if not isinstance(response, dict):
            raise TypeError(f"invalid CacheBlend RPC response {type(response)!r}")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "CacheBlend lookup RPC failed"))
        return response

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.send(message)
        return self.receive()


class TensorParallelLookup:
    """Matcher-rank lookup followed by exact per-rank validation and pinning."""

    def __init__(self, uris: Iterable[str], timeout_ms: int = 10_000):
        self.clients = tuple(LookupClient(uri, timeout_ms) for uri in uris)
        if not self.clients:
            raise ValueError("at least one TP lookup endpoint is required")

    @staticmethod
    def _request_all(
        clients: Iterable[LookupClient],
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Send to every rank before receiving, allowing rank work to overlap."""
        selected = tuple(clients)
        try:
            for client in selected:
                client.send(message)
            return [client.receive() for client in selected]
        except Exception:
            # A REQ socket with an outstanding or failed request cannot be
            # reused. Reset the whole group to restore a consistent state.
            for client in selected:
                client.reset()
            raise

    def lookup_and_prefetch(
        self,
        request_id: str,
        model_scope: str,
        tokens: list[int],
        start_offset: int,
    ) -> tuple[BlendSegment, ...]:
        # Native APC already owns the prefix. Sending only the suffix avoids
        # serializing tens of thousands of irrelevant token IDs on every
        # scheduler lookup. A NumPy payload is pickled as one binary buffer.
        token_suffix = np.asarray(tokens[start_offset:], dtype=np.int64)
        matched = self.clients[0].request(
            {
                "op": "MATCH",
                "model_scope": model_scope,
                "tokens": token_suffix,
                "start_offset": 0,
            }
        )
        candidates = tuple(
            BlendSegment(
                segment.segment_id,
                segment.target_start + start_offset,
            )
            for segment in (
                BlendSegment.from_dict(value) for value in matched.get("segments", [])
            )
        )
        if not candidates:
            return ()

        common = set(candidates)
        segment_payload = [value.to_dict() for value in candidates]
        # Rank 0 just produced the candidates. Other ranks validate in
        # parallel; PREFETCH below atomically revalidates and pins every rank.
        responses = self._request_all(
            self.clients[1:],
            {"op": "VALIDATE", "segments": segment_payload},
        )
        for response in responses:
            common.intersection_update(
                BlendSegment.from_dict(value) for value in response.get("segments", [])
            )
        final = tuple(sorted(common, key=lambda segment: segment.target_start))
        if not final:
            return ()

        try:
            responses = self._request_all(
                self.clients,
                {
                    "op": "PREFETCH",
                    "request_id": request_id,
                    "segments": [value.to_dict() for value in final],
                },
            )
            pinned_sets = [
                {BlendSegment.from_dict(value) for value in response.get("segments", [])}
                for response in responses
            ]
            pinned_common = set(final)
            for pinned in pinned_sets:
                pinned_common.intersection_update(pinned)
            if pinned_common != set(final):
                self.cancel(request_id)
                return ()
            return final
        except Exception:
            self.cancel(request_id)
            raise

    def cancel(self, request_id: str) -> None:
        try:
            self._request_all(
                self.clients,
                {"op": "CANCEL", "request_id": request_id},
            )
        except (RuntimeError, TypeError, zmq.ZMQError):
            logger.warning("Failed to release CacheBlend request %s", request_id)

    def close(self) -> None:
        for client in self.clients:
            client.close()
