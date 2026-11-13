"""Loopback-only UI preview; explicit --live delegates to Sebastian's adapter."""
from __future__ import annotations

import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = Path(__file__).resolve().parent / "public"
FIELDS = ("ProductCategory", "ProductName", "ProductBrand", "ProductDescription", "ProductContents")
MAX_ROWS = 250
MAX_BYTES = 2 * 1024 * 1024


def read_examples():
    directory = ROOT / "docs" / "integration"
    return {kind: json.loads((directory / f"inference_adapter_{kind}.json").read_text(encoding="utf-8"))
            for kind in ("request", "response")}


def validate_request(body):
    if not isinstance(body, dict) or set(body) - {"schema_version", "products"}:
        raise ValueError("Send an object containing products and schema_version.")
    if body.get("schema_version") != "catalogiq-inference-request-v1":
        raise ValueError("Unsupported request format.")
    rows = body.get("products")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError(f"Choose between 1 and {MAX_ROWS} products.")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"record_id", "quarantined", *FIELDS}:
            raise ValueError("Send only the supported product fields.")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in ids:
            raise ValueError("Each product needs a unique, nonempty text ID.")
        ids.add(record_id)
        if any(field not in row or (row[field] is not None and not isinstance(row[field], str)) for field in FIELDS):
            raise ValueError("All five product fields must be text or null.")
        if not isinstance(row.get("quarantined", False), bool):
            raise ValueError("quarantined must be true or false.")
    return rows


def demo_response(body, examples):
    """Only exact supplied examples receive saved predictions; never simulate inference."""
    rows = validate_request(body)
    response = copy.deepcopy(examples["response"])
    output = []
    for row in rows:
        index = next((i for i, example in enumerate(examples["request"]["products"])
                      if all(row[f] == example[f] for f in FIELDS) and not row.get("quarantined", False)), None)
        if index is not None:
            product = copy.deepcopy(examples["response"]["products"][index])
            product["record_id"] = row["record_id"]
        else:
            product = {"record_id": row["record_id"], "review_required": True, "predictions": {}}
            for target in examples["response"]["products"][0]["predictions"]:
                product["predictions"][target] = {
                    "status": "unavailable", "selection_status": "deferred" if target == "Platform" else "selected",
                    "predicted_label": None, "confidence": None, "raw_confidence": None,
                    "probabilities": None, "calibrated_probabilities": None, "calibrated_confidence": None,
                    "label_mapping": {}, "review_required": True,
                    "review_reasons": ["target_unavailable" if target == "Platform" else "models_not_connected"],
                    "lookup_status": None, "provenance": {}, "error": None}
        output.append(product)
    response["products"] = output
    return response


def create_server(port=8765, *, engine=None, infer=None):
    examples = read_examples()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status, body):
            self.send_bytes(status, json.dumps(body, allow_nan=False).encode(), "application/json; charset=utf-8")

        def allowed_host(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in expected

        def do_GET(self):
            if not self.allowed_host():
                return self.send_json(403, {"error": "Use the local preview address."})
            path = urlsplit(self.path).path
            if path == "/api/health":
                return self.send_json(200, {"mode": "live" if engine else "demo", "max_rows": MAX_ROWS, "max_bytes": MAX_BYTES})
            if path == "/api/examples":
                return self.send_json(200, examples)
            target = (PUBLIC / (unquote(path).lstrip("/") or "index.html")).resolve()
            if not target.is_relative_to(PUBLIC) or not target.is_file():
                return self.send_json(404, {"error": "Page not found."})
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".js":
                mime = "text/javascript"
            self.send_bytes(200, target.read_bytes(), mime)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.allowed_host() or (origin and origin not in {
                f"http://localhost:{self.server.server_port}", f"http://127.0.0.1:{self.server.server_port}"}):
                return self.send_json(403, {"error": "Open this app from its local address."})
            if self.path != "/api/predict":
                return self.send_json(404, {"error": "Endpoint not found."})
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self.send_json(415, {"error": "Send a JSON request."})
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_json(400, {"error": "Invalid request length."})
            if not 0 < size <= MAX_BYTES:
                return self.send_json(413, {"error": "This batch is too large. Use a file under 2 MB."})
            try:
                body = json.loads(self.rfile.read(size))
                validate_request(body)
                response = infer(engine, body) if engine else demo_response(body, examples)
            except (ValueError, UnicodeError):
                return self.send_json(400, {"error": "Check product IDs and the five supported text fields, then try again."})
            except Exception:
                self.log_error("Prediction request failed")
                return self.send_json(503, {"error": "Classification is unavailable. Your products are still here; try again later."})
            self.send_json(200, response)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    engine = infer = None
    if args.live:
        if args.artifacts is None:
            parser.error("--live requires --artifacts pointing to the pinned serving files")
        pins = json.loads((ROOT / "src/catalogiq/resources/inference/frozen_routes.json").read_text())
        missing = [name for route in pins["routes"].values() for name in route["files_sha256"]
                   if not (args.artifacts / name).is_file()]
        if missing:
            parser.error(f"Missing {len(missing)} serving files. First: {missing[0]}")
        sys.path.insert(0, str(ROOT / "src"))
        from catalogiq.inference_engine import InferenceEngine
        from catalogiq.inference_adapter import infer_request
        engine, infer = InferenceEngine(args.artifacts), infer_request
    server = create_server(args.port, engine=engine, infer=infer)
    print(f"CatalogIQ {'live' if engine else 'demo'}: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
