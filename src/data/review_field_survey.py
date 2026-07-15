from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


APP_DIR = Path(__file__).with_name("review_app")
DECISION_ACTIONS = {"accept", "reject", "replace"}


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).casefold()).strip()


def _group_id(crop: str, disease: str) -> str:
    value = f"{_key(crop)}\0{_key(disease)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def _canonical_labels(manifest: dict[str, Any]) -> list[str]:
    labels = set()
    taxonomy = manifest.get("canonical_taxonomy", {}).get("crops", {})
    for diseases in taxonomy.values():
        labels.update(_text(label) for label in diseases if _text(label))
    for record in manifest.get("records", []):
        canonical = record.get("normalization", {}).get("disease", {}).get("canonical")
        if canonical:
            labels.add(_text(canonical))
    return sorted(labels)


def _suggest(label: str, canonical_labels: list[str]) -> tuple[str | None, float | None]:
    source = _key(label)
    if not source or not canonical_labels:
        return None, None
    ranked = sorted(
        ((SequenceMatcher(None, source, _key(candidate)).ratio(), candidate) for candidate in canonical_labels),
        reverse=True,
    )
    score, candidate = ranked[0]
    return (candidate, round(score, 3)) if score >= 0.48 else (None, None)


def _priority(reasons: set[str], count: int) -> int:
    weights = {
        "missing_disease": 100, "unknown_value": 90, "multiple_crops": 80,
        "compound_or_ambiguous_label": 70, "multilingual_or_transliterated": 60,
        "pest_treatment_or_non_disease": 50, "ambiguous_label": 40, "unknown_crop": 30,
    }
    return max((weights.get(reason, 20) for reason in reasons), default=10) + min(count, 20)


def build_review_groups(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    canonical = _canonical_labels(manifest)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in manifest.get("records", []):
        grouped[(_text(record.get("crop")), _text(record.get("label")))].append(record)

    groups = []
    for (crop, disease), records in grouped.items():
        disease_norm = records[0].get("normalization", {}).get("disease", {})
        crop_norm = records[0].get("normalization", {}).get("crop", {})
        reasons = set(disease_norm.get("review_reasons", [])) | set(crop_norm.get("review_reasons", []))
        suggested = disease_norm.get("canonical")
        confidence = disease_norm.get("confidence")
        if not suggested:
            suggested, confidence = _suggest(disease, canonical)
        symptoms = Counter(
            _text(record.get("metadata", {}).get("_Symptoms"))
            for record in records if _text(record.get("metadata", {}).get("_Symptoms"))
        )
        image_paths = [record.get("image_path") for record in records if record.get("image_path")]
        groups.append({
            "group_id": _group_id(crop, disease),
            "crop": crop,
            "canonical_crop": crop_norm.get("canonical"),
            "original_disease": disease,
            "suggested_disease": suggested,
            "confidence": confidence,
            "review_reasons": sorted(reasons),
            "symptoms": [value for value, _ in symptoms.most_common(5)],
            "record_ids": [record.get("record_id") for record in records],
            "record_count": len(records),
            "image_paths": image_paths[:12],
            "requires_manual_review": disease_norm.get("status") == "manual_review" or crop_norm.get("status") == "manual_review",
            "priority": _priority(reasons, len(records)),
        })

    # Similar labels are adjacent while risk and review status still dominate ordering.
    groups.sort(key=lambda item: (
        not item["requires_manual_review"], -item["priority"],
        _key(item["suggested_disease"] or item["original_disease"]), _key(item["crop"]),
    ))
    for group in groups:
        seed = _key(group["suggested_disease"] or group["original_disease"])
        group["similarity_group"] = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return groups


class ReviewStore:
    def __init__(self, manifest_path: Path, output_path: Path, decisions_path: Path):
        self.manifest_path = manifest_path.resolve()
        self.output_path = output_path.resolve()
        self.decisions_path = decisions_path.resolve()
        self.lock = threading.Lock()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.groups = build_review_groups(self.manifest)
        self.group_by_id = {group["group_id"]: group for group in self.groups}
        self.canonical_labels = _canonical_labels(self.manifest)
        self.history = self._load_history()
        self.latest = {event["group_id"]: event for event in self.history}
        self._write_validated_manifest()

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.decisions_path.exists():
            return []
        history = []
        for line_number, line in enumerate(self.decisions_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("group_id") in self.group_by_id:
                history.append(event)
            else:
                raise ValueError(f"Unknown group_id in decisions file at line {line_number}")
        return history

    def summary(self) -> dict[str, int]:
        decisions = Counter(event["action"] for event in self.latest.values())
        validated_records = sum(
            self.group_by_id[group_id]["record_count"]
            for group_id, event in self.latest.items() if event["action"] in {"accept", "replace"}
        )
        return {
            "groups": len(self.groups), "pending_groups": len(self.groups) - len(self.latest),
            "accepted_groups": decisions["accept"], "replaced_groups": decisions["replace"],
            "rejected_groups": decisions["reject"], "validated_records": validated_records,
            "total_records": len(self.manifest.get("records", [])),
        }

    def queue(self, status: str = "pending", search: str = "") -> dict[str, Any]:
        needle = _key(search)
        items = []
        for group in self.groups:
            decision = self.latest.get(group["group_id"])
            current = decision["action"] if decision else "pending"
            if status != "all" and current != status:
                continue
            if needle and needle not in _key(f"{group['crop']} {group['original_disease']} {group['suggested_disease']}"):
                continue
            item = dict(group)
            item["decision"] = decision
            item["image_urls"] = [f"/image?path={quote(path, safe='')}" for path in group["image_paths"]]
            items.append(item)
        return {"summary": self.summary(), "canonical_labels": self.canonical_labels, "groups": items}

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        group_id = _text(payload.get("group_id"))
        action = _text(payload.get("action")).casefold()
        reviewer = _text(payload.get("reviewer"))
        replacement = _text(payload.get("canonical_disease"))
        note = _text(payload.get("note"))
        if group_id not in self.group_by_id:
            raise ValueError("Unknown review group")
        if action not in DECISION_ACTIONS:
            raise ValueError("Action must be accept, reject, or replace")
        group = self.group_by_id[group_id]
        if action == "accept" and not group.get("suggested_disease"):
            raise ValueError("This group has no suggestion to accept")
        if action == "replace" and not replacement:
            raise ValueError("A replacement canonical disease is required")
        canonical = group.get("suggested_disease") if action == "accept" else replacement if action == "replace" else None
        event = {
            "event_id": hashlib.sha256(os.urandom(32)).hexdigest()[:20],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "group_id": group_id, "record_ids": group["record_ids"], "action": action,
            "reviewer": reviewer or "anonymous", "canonical_disease": canonical,
            "original_disease": group["original_disease"], "suggested_disease": group.get("suggested_disease"),
            "suggestion_confidence": group.get("confidence"), "note": note,
        }
        with self.lock:
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with self.decisions_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            self.history.append(event)
            self.latest[group_id] = event
            self._write_validated_manifest()
        return {"decision": event, "summary": self.summary()}

    def _write_validated_manifest(self) -> None:
        record_to_group = {
            record_id: group["group_id"] for group in self.groups for record_id in group["record_ids"]
        }
        records = []
        for source_record in self.manifest.get("records", []):
            record = dict(source_record)
            group_id = record_to_group.get(record.get("record_id"))
            event = self.latest.get(group_id)
            accepted = bool(event and event["action"] in {"accept", "replace"})
            record["validation"] = {
                "group_id": group_id, "status": "validated" if accepted else event["action"] if event else "pending",
                "canonical_disease": event.get("canonical_disease") if accepted else None,
                "eligible_for_training": accepted,
                "latest_decision_event_id": event.get("event_id") if event else None,
            }
            records.append(record)
        output = {
            "schema_version": "1.2", "dataset_name": self.manifest.get("dataset_name", "field_survey"),
            "created_at": datetime.now(timezone.utc).isoformat(), "source_manifest": self.manifest_path.as_posix(),
            "validation_policy": {
                "training_eligibility": "Only records with validation.eligible_for_training=true may be used.",
                "uncertain_suggestions_auto_applied": False, "records_discarded": False,
            },
            "statistics": self.summary(), "audit_history": self.history, "records": records,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temp.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, self.output_path)


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/queue":
            query = parse_qs(parsed.query)
            self._json(self.store.queue(query.get("status", ["pending"])[0], query.get("search", [""])[0]))
            return
        if parsed.path == "/image":
            requested = Path(unquote(parse_qs(parsed.query).get("path", [""])[0])).resolve()
            allowed = {Path(path).resolve() for group in self.store.groups for path in group["image_paths"]}
            if requested not in allowed:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._file(requested)
            return
        asset = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        path = (APP_DIR / asset).resolve()
        if APP_DIR.resolve() not in path.parents:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self._file(path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/decisions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100_000:
                raise ValueError("Request is too large")
            result = self.store.decide(json.loads(self.rfile.read(length) or b"{}"))
            self._json(result, HTTPStatus.CREATED)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def serve(manifest: Path, output: Path, decisions: Path, host: str, port: int) -> None:
    ReviewHandler.store = ReviewStore(manifest, output, decisions)
    server = ThreadingHTTPServer((host, port), ReviewHandler)
    print(f"Field survey review: http://{host}:{port}")
    print(f"Validated manifest: {output}")
    print(f"Audit log: {decisions}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the human validation interface for the field survey dataset.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/field_survey/cleaned_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/field_survey/validated_manifest.json"))
    parser.add_argument("--decisions", type=Path, default=Path("data/manifests/field_survey/review_decisions.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.manifest, args.output, args.decisions, args.host, args.port)


if __name__ == "__main__":
    main()
