"""Frozen student-level split for G6P Eedi prospective study."""
from __future__ import annotations

from hashlib import sha256
from typing import Iterable

DATASET_ID = "eedi_neurips_2020"
SPLIT_SEED = "WDLF-G6P-SPLIT-SEED-v1"
SPLIT_PROPORTIONS = (0.60, 0.20, 0.20)  # development, calibration, confirmatory


def normalize_user_id(user_id: str | int) -> str:
    return str(user_id).strip()


def assign_eedi_split(user_id: str | int) -> str:
    material = f"{DATASET_ID}{normalize_user_id(user_id)}{SPLIT_SEED}".encode("utf-8")
    digest = sha256(material).hexdigest()
    u = int(digest[:12], 16) / float(16**12)
    dev, cal, _ = SPLIT_PROPORTIONS
    if u < dev:
        return "development"
    if u < dev + cal:
        return "calibration"
    return "confirmatory"


def assign_many(user_ids: Iterable[str | int]) -> dict[str, str]:
    return {normalize_user_id(u): assign_eedi_split(u) for u in user_ids}
