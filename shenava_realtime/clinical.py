"""Optional deterministic NER → rules → JSON/SQLite, outside the ASR worker.

Entities are mentions, NOT diagnoses. Values retain their dictated units;
no inferred conversions, drug-dose associations, or clinical recommendations.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from .fst import TrieFST
from .text_normalize import digits_to_ascii

logger = logging.getLogger(__name__)
DEFAULT_ENTITIES = {
    "CABG": "procedure", "PCI": "procedure", "ECG": "test", "MRI": "test",
    "CT": "test", "CBC": "test", "HbA1c": "test", "FBS": "test",
    "BP": "vital", "SpO2": "vital", "تب": "symptom", "درد قفسه سینه": "symptom",
    "تنگی نفس": "symptom", "دیابت": "condition", "CKD": "condition",
    "MI": "condition", "DVT": "condition", "آسپرین": "drug", "متفورمین": "drug",
    "انسولین": "drug", "IV": "route", "IM": "route", "PO": "route",
}


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int
    assertion: str = "unspecified"


class DictionaryNER:
    def __init__(self, terms: Optional[Mapping[str, str]] = None) -> None:
        self.trie = TrieFST()
        self.trie.add_many(DEFAULT_ENTITIES if terms is None else terms)

    def extract(self, text: str) -> list[Entity]:
        tokens = list(re.finditer(r"[\w\u200c]+", text))
        entities = []
        # Punctuation is a hard phrase boundary, never match through it.
        groups = []
        group = []
        for token in tokens:
            if group and text[group[-1].end():token.start()].strip():
                groups.append(group)
                group = []
            group.append(token)
        if group:
            groups.append(group)
        for group in groups:
            for label, start, end in self.trie.rewrite_spans([t.group() for t in group]):
                if label is not None:
                    a, b = group[start].start(), group[end - 1].end()
                    entities.append(Entity(text[a:b], label, a, b))
        return entities


def extract_record(text: str, ner: Optional[DictionaryNER] = None) -> dict:
    entities = (ner or DictionaryNER()).extract(text)
    mentions = []
    for entity in entities:
        # Only explicit adjacent negation, bounded by sentence punctuation.
        suffix = text[entity.end:entity.end + 32]
        assertion = "negated" if re.match(r"\s+(?:ندارد|نیست|رد شد)(?:\W|$)", suffix) else "unspecified"
        mentions.append({**asdict(entity), "assertion": assertion})
    measurements = []
    # No plausibility-based suppression: abnormal values remain reviewable.
    pattern = r"(?<![\w.])(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(mmHg|mg/dL|mmol/L|mcg/min|mg/min|mL/h|mg|mcg|mL|kg|bpm|rpm|°C|°F|%)(?!\w)"
    for match in re.finditer(pattern, text):
        measurements.append({"value": digits_to_ascii(match[1]), "unit": match[2],
                             "start": match.start(), "end": match.end()})
    for match in re.finditer(r"\bBP\s+(\d{2,3})/(\d{2,3})(?!\d)", text):
        measurements.append({"kind": "blood_pressure", "systolic": int(match[1]),
                             "diastolic": int(match[2]), "unit": None,
                             "start": match.start(), "end": match.end()})
    return {"schema_version": 1, "text": text, "entities": mentions,
            "measurements": measurements, "review_required": True}


class ClinicalWorker:
    """Opt-in bounded sink; all disk I/O and extraction run on its own thread.

    Overflow is reported, never blocks audio. SQLite transactions store exactly
    one completed utterance each. Files contain sensitive health information:
    protect them using filesystem permissions/encrypted storage and retention.
    """
    def __init__(self, sqlite_path: Optional[str] = None, jsonl_path: Optional[str] = None,
                 max_pending: int = 32) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.sqlite_path, self.jsonl_path = sqlite_path, jsonl_path
        self.queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self.thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.error: Optional[Exception] = None
        self.ready = threading.Event()
        self.stopping = threading.Event()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.ready.clear()
        self.stopping.clear()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="clinical-output", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("Clinical storage startup timed out")
        if self.error:
            raise RuntimeError("Clinical storage startup failed") from self.error

    def submit(self, text: str) -> bool:
        if self.stopping.is_set() or self.error or not self.thread or not self.thread.is_alive():
            return False
        if len(text) > 20000:
            raise ValueError("Clinical input exceeds utterance limit")
        try:
            self.queue.put_nowait(text)
            return True
        except queue.Full:
            self.dropped += 1
            logger.error("Clinical queue full; record not stored (total %d)", self.dropped)
            return False

    def stop(self) -> None:
        self.stopping.set()
        if self.thread:
            self.thread.join(5)
            if self.thread.is_alive():
                logger.error("Clinical output still draining")

    def _run(self) -> None:
        db = output = None
        try:
            if self.sqlite_path:
                db = sqlite3.connect(self.sqlite_path, timeout=2)
                db.execute("CREATE TABLE IF NOT EXISTS utterances (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, record_json TEXT NOT NULL)")
                db.commit()
            if self.jsonl_path:
                output = Path(self.jsonl_path).open("a", encoding="utf-8")
            ner = DictionaryNER()
            self.ready.set()
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    text = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                record = extract_record(text, ner)
                record["created_at"] = datetime.now(timezone.utc).isoformat()
                payload = json.dumps(record, ensure_ascii=False)
                if db:
                    with db:
                        db.execute("INSERT INTO utterances(created_at, record_json) VALUES (?, ?)",
                                   (record["created_at"], payload))
                if output:
                    output.write(payload + "\n")
                    output.flush()
        except Exception as exc:
            self.error = exc
            logger.exception("Clinical output failed; ASR remains independent")
        finally:
            self.ready.set()
            if db:
                db.close()
            if output:
                output.close()
