"""
Sanity-check report over a processed manifest: split sizes, class balance
per split, and average crops-per-source. Run this before training — a
silently imbalanced or leaked split will quietly wreck model evaluation.
"""
import csv
from collections import Counter
from pathlib import Path


def summarize(processed_manifest_path: Path) -> dict:
    with open(processed_manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"total_crops": 0}

    report: dict = {"total_crops": len(rows)}

    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        label_counts = Counter(r["label"] for r in split_rows)
        source_count = len({r["source_id"] for r in split_rows})
        report[split] = {
            "crops": len(split_rows),
            "sources": source_count,
            "label_counts": dict(label_counts),
        }

    # Leakage check: no source_id should appear in more than one split.
    source_to_splits: dict[str, set] = {}
    for r in rows:
        source_to_splits.setdefault(r["source_id"], set()).add(r["split"])
    leaked = {sid: splits for sid, splits in source_to_splits.items() if len(splits) > 1}
    report["leakage_check"] = "PASS" if not leaked else f"FAIL: {len(leaked)} source(s) span multiple splits"

    return report


def print_report(processed_manifest_path: Path) -> None:
    report = summarize(processed_manifest_path)
    print(f"\n=== Dataset report: {processed_manifest_path} ===")
    print(f"Total face crops: {report.get('total_crops', 0)}")
    for split in ("train", "val", "test"):
        if split in report:
            s = report[split]
            print(f"  {split:5s}: {s['crops']:6d} crops | {s['sources']:4d} sources | labels={s['label_counts']}")
    print(f"Leakage check: {report.get('leakage_check', 'N/A')}\n")
