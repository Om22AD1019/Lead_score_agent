"""
Lightweight storage for lead scoring results.
Uses a local JSON file so the whole agent runs with zero external DB setup.
"""

import json
import os
import threading
from typing import List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "leads_store.json")
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