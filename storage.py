"""
Lightweight storage for lead scoring results.
Uses local JSON files — no database needed.

Files:
  leads_store.json    -> all scored leads
  ai_summaries.json   -> all AI-generated summaries
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict

DB_PATH      = os.path.join(os.path.dirname(__file__), "leads_store.json")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "ai_summaries.json")

_lock = threading.Lock()



def _read_all() -> List[Dict]:
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_result(result: dict) -> dict:
    with _lock:
        data = _read_all()
        result["record_id"] = len(data) + 1
        data.append(result)
        with open(DB_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return result


def get_all() -> List[Dict]:
    with _lock:
        return _read_all()


def get_by_id(record_id: int) -> Dict | None:
    with _lock:
        data = _read_all()
        for row in data:
            if row.get("record_id") == record_id:
                return row
        return None



def _read_summaries() -> List[Dict]:
    if not os.path.exists(SUMMARY_PATH):
        return []
    with open(SUMMARY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_summary(record_id: int, lead_id: str, name: str, summary_text: str) -> dict:
    """Save an AI-generated summary to ai_summaries.json."""
    with _lock:
        summaries = _read_summaries()
        entry = {
            "summary_id":  len(summaries) + 1,
            "record_id":   record_id,
            "lead_id":     lead_id,
            "name":        name,
            "summary":     summary_text,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        summaries.append(entry)
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summaries, f, indent=2)
        return entry


def get_all_summaries() -> List[Dict]:
    with _lock:
        return _read_summaries()


def get_summaries_by_lead(record_id: int) -> List[Dict]:
    """Get all summaries ever generated for a specific lead."""
    with _lock:
        return [s for s in _read_summaries()
                if s.get("record_id") == record_id]