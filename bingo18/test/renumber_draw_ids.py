"""Sort the Bingo18 CSV chronologically and regenerate sequential draw IDs."""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from crawler import (
    BATCH_DAYS,
    CSV_FIELDNAMES,
    REQUEST_DELAY,
    create_session,
    fetch_batch,
    generate_date_batches,
    validate_draw,
)


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "bingo18.csv"


def load_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read and validate every Bingo18 row without changing its source ID."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDNAMES:
            raise ValueError(
                f"Unexpected CSV header: {reader.fieldnames!r}; "
                f"expected {CSV_FIELDNAMES!r}"
            )

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in CSV_FIELDNAMES):
                raise ValueError(f"Malformed CSV row at line {line_number}")
            record = {field: row[field] for field in CSV_FIELDNAMES}
            validate_draw(record)
            source_id = str(record["draw_id"])
            if source_id in seen_ids:
                raise ValueError(
                    f"Duplicate draw_id/source ID {source_id!r} at line {line_number}"
                )
            seen_ids.add(source_id)
            record.update({
                "n1": int(record["n1"]),
                "n2": int(record["n2"]),
                "n3": int(record["n3"]),
                "sum": int(record["sum"]),
            })
            records.append(record)
    return records


def fetch_timestamps(
    records: list[dict[str, Any]],
) -> dict[str, datetime]:
    """Look up source timestamps so legacy API IDs can be sorted chronologically."""
    wanted_ids = {str(record["draw_id"]) for record in records}
    timestamps: dict[str, datetime] = {}
    api_values: dict[str, tuple[int, int, int, int]] = {}
    session = create_session()
    try:
        batches = list(generate_date_batches(date(2025, 7, 27), date.today(), BATCH_DAYS))
        for batch_index, (batch_start, batch_end) in enumerate(batches, start=1):
            print(
                f"Timestamp lookup {batch_index}/{len(batches)}: "
                f"{batch_start} -> {batch_end}"
            )
            batch, _, invalid = fetch_batch(session, batch_start, batch_end)
            if invalid:
                print(
                    f"Warning: skipped {invalid} invalid API record(s) "
                    f"in {batch_start} -> {batch_end}",
                    file=sys.stderr,
                )
            for draw in batch:
                source_id = str(draw["source_id"])
                if source_id not in wanted_ids:
                    continue
                timestamp = draw["_sort_datetime"]
                previous = timestamps.get(source_id)
                values = (
                    int(draw["n1"]),
                    int(draw["n2"]),
                    int(draw["n3"]),
                    int(draw["sum"]),
                )
                if previous is not None and (
                    previous != timestamp or api_values[source_id] != values
                ):
                    raise ValueError(
                        f"Conflicting API records for source ID {source_id}"
                    )
                timestamps[source_id] = timestamp
                api_values[source_id] = values
            if batch_index < len(batches):
                time.sleep(REQUEST_DELAY)
    finally:
        session.close()

    missing = wanted_ids - timestamps.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"API did not return timestamps for {len(missing)} CSV row(s); "
            f"sample source IDs: {sample}"
        )
    for record in records:
        source_id = str(record["draw_id"])
        record_values = (
            int(record["n1"]),
            int(record["n2"]),
            int(record["n3"]),
            int(record["sum"]),
        )
        if record_values != api_values[source_id]:
            raise ValueError(
                f"CSV values conflict with API data for source ID {source_id}"
            )
    return timestamps


def chronological_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order legacy source-ID rows using API timestamps, or retain generated order."""
    if all(record["draw_id"] == str(index) for index, record in enumerate(records, 1)):
        return records

    timestamps = fetch_timestamps(records)
    ordered = sorted(
        records,
        key=lambda record: (
            timestamps[str(record["draw_id"])],
            str(record["draw_id"]),
        ),
    )
    return ordered


def write_csv_atomic(records: list[dict[str, Any]], csv_path: Path) -> None:
    """Write the requested temporary CSV, then replace the original atomically."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_name("bingo18.csv.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for draw_id, record in enumerate(records, start=1):
                writer.writerow({
                    "draw_id": draw_id,
                    "n1": record["n1"],
                    "n2": record["n2"],
                    "n3": record["n3"],
                    "sum": record["sum"],
                })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    try:
        records = load_csv(CSV_PATH)
        ordered = chronological_records(records)
        for record in ordered:
            validate_draw(record)
        write_csv_atomic(ordered, CSV_PATH)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Renumbering failed; original CSV was not replaced: {exc}", file=sys.stderr)
        return 1

    print(f"Rows processed: {len(ordered)}")
    print("First draw_id: 1" if ordered else "First draw_id: N/A")
    print(f"Last draw_id: {len(ordered)}" if ordered else "Last draw_id: N/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
