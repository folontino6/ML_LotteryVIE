"""Download and maintain historical Vietlott Bingo18 draw results."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://ketquasoxo.top/api/gbingo/draws/by-date"
START_DATE = date(2025, 7, 27)
BATCH_DAYS = 30
REQUEST_TIMEOUT = 20
MAX_RETRIES = 4
REQUEST_DELAY = 0.5
NUMBER_MIN = 1
NUMBER_MAX = 6
CSV_FIELDNAMES = ["draw_id", "n1", "n2", "n3", "sum"]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "data" / "bingo18.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bingo18-crawler")


def create_session() -> requests.Session:
    """Create an HTTP session with bounded retries for transient failures."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def generate_date_batches(
    start_date: date,
    end_date: date,
    batch_days: int = BATCH_DAYS,
) -> Iterator[tuple[date, date]]:
    """Yield inclusive adjacent date ranges without gaps or overlaps."""
    if batch_days < 1:
        raise ValueError("batch_days must be positive")
    current = start_date
    while current <= end_date:
        batch_end = min(current + timedelta(days=batch_days - 1), end_date)
        yield current, batch_end
        current = batch_end + timedelta(days=1)


def parse_draw(raw: Any) -> dict[str, Any]:
    """Parse one API draw, retaining its source identity and timestamp internally."""
    if not isinstance(raw, dict):
        raise ValueError("draw record is not a JSON object")

    source_id = raw.get("_id")
    if source_id is None or not str(source_id).strip():
        raise ValueError("missing draw _id")
    source_id = str(source_id).strip()

    result = raw.get("winningResult")
    if isinstance(result, bool) or not isinstance(result, (str, int)):
        raise ValueError(f"invalid winningResult for draw {source_id}: {result!r}")
    result_text = str(result)
    if not re.fullmatch(r"\d{3}", result_text):
        raise ValueError(f"winningResult must contain exactly three digits: {result!r}")
    numbers = [int(digit) for digit in result_text]
    if any(number < NUMBER_MIN or number > NUMBER_MAX for number in numbers):
        raise ValueError(
            f"draw {source_id} contains a number outside "
            f"{NUMBER_MIN}-{NUMBER_MAX}: {numbers}"
        )

    calculated_sum = sum(numbers)
    api_sum = raw.get("sum")
    if api_sum is not None:
        try:
            provided_sum = int(api_sum)
        except (TypeError, ValueError):
            logger.warning(
                "Draw %s has a non-numeric API sum %r; using calculated sum %d",
                source_id, api_sum, calculated_sum,
            )
        else:
            if provided_sum != calculated_sum:
                logger.warning(
                    "Draw %s API sum %d conflicts with calculated sum %d; "
                    "using calculated sum",
                    source_id, provided_sum, calculated_sum,
                )

    draw_at = raw.get("drawAt")
    if not draw_at:
        raise ValueError(f"missing drawAt for draw {source_id}")
    try:
        sort_datetime = datetime.fromisoformat(
            str(draw_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"invalid drawAt for draw {source_id}: {draw_at!r}"
        ) from exc
    if sort_datetime.tzinfo is None:
        raise ValueError(f"drawAt has no timezone for draw {source_id}")

    return {
        "source_id": source_id,
        "n1": numbers[0],
        "n2": numbers[1],
        "n3": numbers[2],
        "sum": calculated_sum,
        "_sort_datetime": sort_datetime,
    }


def validate_draw(draw: dict[str, Any]) -> None:
    """Validate the required Bingo18 fields and their mathematical relationship."""
    identity = draw.get("source_id", draw.get("draw_id", ""))
    if not str(identity).strip():
        raise ValueError("draw identity is empty")
    numbers: list[int] = []
    for field in ("n1", "n2", "n3"):
        value = draw.get(field)
        if isinstance(value, bool):
            raise ValueError(f"{field} is not numeric")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} is not numeric") from exc
        if str(value).strip() != str(number):
            raise ValueError(f"{field} is not an integer: {value!r}")
        if not NUMBER_MIN <= number <= NUMBER_MAX:
            raise ValueError(f"{field} is outside {NUMBER_MIN}-{NUMBER_MAX}")
        numbers.append(number)

    try:
        total = int(draw.get("sum"))
    except (TypeError, ValueError) as exc:
        raise ValueError("sum is not numeric") from exc
    if str(draw.get("sum")).strip() != str(total):
        raise ValueError(f"sum is not an integer: {draw.get('sum')!r}")
    if total != sum(numbers):
        raise ValueError(
            f"sum mismatch: stored {total}, calculated {sum(numbers)}"
        )


def fetch_batch(
    session: requests.Session,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], int, int]:
    """Fetch and validate one date batch; malformed individual draws are skipped."""
    response = session.get(
        API_URL,
        params={
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError(
            f"API returned invalid JSON for {start_date} through {end_date}"
        ) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(
            f"Unexpected API response structure for {start_date} through {end_date}"
        )

    valid: list[dict[str, Any]] = []
    invalid = 0
    for index, raw_draw in enumerate(payload["data"]):
        try:
            draw = parse_draw(raw_draw)
            validate_draw(draw)
        except (TypeError, ValueError) as exc:
            invalid += 1
            source_id = raw_draw.get("_id") if isinstance(raw_draw, dict) else None
            logger.warning(
                "Skipping invalid API record %d (source_id=%r): %s",
                index, source_id, exc,
            )
            continue
        valid.append(draw)
    return valid, len(payload["data"]), invalid


def record_values(draw: dict[str, Any]) -> tuple[int, int, int, int]:
    return int(draw["n1"]), int(draw["n2"]), int(draw["n3"]), int(draw["sum"])


def deduplicate_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate API records by source ID or timestamp, never by CSV index."""
    unique: list[dict[str, Any]] = []
    by_source_id: dict[str, dict[str, Any]] = {}
    by_timestamp: dict[datetime, dict[str, Any]] = {}
    duplicates = 0
    for draw in records:
        source_id = str(draw["source_id"])
        timestamp = draw["_sort_datetime"]
        previous = by_source_id.get(source_id) or by_timestamp.get(timestamp)
        if previous is None:
            unique.append(draw)
            by_source_id[source_id] = draw
            by_timestamp[timestamp] = draw
            continue
        duplicates += 1
        if (
            str(previous["source_id"]) != source_id
            or record_values(previous) != record_values(draw)
            or previous["_sort_datetime"] != timestamp
        ):
            logger.warning(
                "Conflicting API records for source_id %s at %s; "
                "keeping source_id %s values %s",
                source_id, timestamp.isoformat(), previous["source_id"],
                record_values(previous),
            )
    return unique, duplicates


def load_existing_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load valid records from an existing Bingo18 CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDNAMES:
            raise ValueError(
                f"Unexpected CSV header in {csv_path}: {reader.fieldnames!r}; "
                f"expected {CSV_FIELDNAMES!r}"
            )
        records: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            draw = {field: row.get(field, "") for field in CSV_FIELDNAMES}
            try:
                validate_draw(draw)
                draw.update({
                    "n1": int(draw["n1"]),
                    "n2": int(draw["n2"]),
                    "n3": int(draw["n3"]),
                    "sum": int(draw["sum"]),
                })
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid existing CSV row %d in %s: %s",
                    line_number, csv_path, exc,
                )
                continue
            records.append(draw)
    return records


def merge_records(
    existing: list[dict[str, Any]],
    downloaded: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Reconcile stored rows against the source-identified chronological history."""
    unique, duplicates = deduplicate_records(downloaded)
    unique.sort(key=lambda record: (record["_sort_datetime"], record["source_id"]))

    overlap = min(len(existing), len(unique))
    for index in range(overlap):
        if record_values(existing[index]) != record_values(unique[index]):
            logger.warning(
                "Existing CSV row %d differs from current API chronology; "
                "rebuilding it from source_id %s at %s",
                index + 1,
                unique[index]["source_id"],
                unique[index]["_sort_datetime"].isoformat(),
            )
    if len(existing) > len(unique):
        logger.warning(
            "Existing CSV has %d row(s) beyond current API history; "
            "the refreshed API dataset will be used",
            len(existing) - len(unique),
        )
    return unique, duplicates


def save_csv_atomic(records: list[dict[str, Any]], csv_path: Path) -> None:
    """Write sequential draw numbers and atomically replace the CSV."""
    for draw in records:
        validate_draw(draw)
    records.sort(key=lambda record: (record["_sort_datetime"], record["source_id"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=csv_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for draw_number, draw in enumerate(records, start=1):
                writer.writerow({
                    "draw_id": draw_number,
                    "n1": draw["n1"],
                    "n2": draw["n2"],
                    "n3": draw["n3"],
                    "sum": draw["sum"],
                })
        os.replace(temp_path, csv_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=START_DATE,
        help=f"First date to crawl (default: {START_DATE.isoformat()})",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Last date to crawl (default: today's local date)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end_date = args.end_date or date.today()
    if args.start_date > end_date:
        logger.error("Start date %s is after end date %s", args.start_date, end_date)
        return 2

    csv_path = (args.output_dir or OUTPUT_CSV.parent) / OUTPUT_CSV.name
    try:
        existing = load_existing_csv(csv_path)
    except (OSError, csv.Error, ValueError) as exc:
        logger.error("Cannot load existing CSV %s: %s", csv_path, exc)
        return 1

    logger.info("Bingo18 crawler")
    logger.info("Range: %s -> %s", args.start_date, end_date)
    logger.info("Output: %s", csv_path)
    logger.info(
        "Refreshing the complete API range; loaded %d existing valid record(s)",
        len(existing),
    )

    downloaded: list[dict[str, Any]] = []
    total_api_records = total_invalid = 0
    session = create_session()
    try:
        batches = list(generate_date_batches(args.start_date, end_date))
        for batch_index, (batch_start, batch_end) in enumerate(batches, start=1):
            logger.info(
                "Batch %d/%d: %s -> %s",
                batch_index, len(batches), batch_start, batch_end,
            )
            batch_records, received, invalid = fetch_batch(
                session, batch_start, batch_end
            )
            total_api_records += received
            total_invalid += invalid
            downloaded.extend(batch_records)
            logger.info(
                "Received: %d; valid: %d; invalid: %d",
                received, len(batch_records), invalid,
            )
            if batch_index < len(batches):
                time.sleep(REQUEST_DELAY)
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.error(
            "Crawl failed before saving; existing CSV was left unchanged: %s", exc
        )
        return 1
    finally:
        session.close()

    unique_downloaded, batch_duplicates = deduplicate_records(downloaded)
    merged, merge_duplicates = merge_records(existing, unique_downloaded)
    try:
        save_csv_atomic(merged, csv_path)
    except (OSError, csv.Error, ValueError) as exc:
        logger.error("Could not save CSV %s: %s", csv_path, exc)
        return 1

    dated_records = [
        draw for draw in unique_downloaded if draw.get("_sort_datetime") is not None
    ]
    if dated_records:
        earliest = min(dated_records, key=lambda draw: draw["_sort_datetime"])
        latest = max(dated_records, key=lambda draw: draw["_sort_datetime"])
        logger.info(
            "Retrieved draw range: %s -> %s",
            earliest["_sort_datetime"].isoformat(),
            latest["_sort_datetime"].isoformat(),
        )

    logger.info("Finished")
    logger.info("Total API records: %d", total_api_records)
    logger.info("Valid unique downloaded records: %d", len(unique_downloaded))
    logger.info("Invalid records skipped: %d", total_invalid)
    logger.info(
        "Duplicates removed: %d", batch_duplicates + merge_duplicates
    )
    logger.info("Final CSV records: %d", len(merged))
    logger.info("Final draw_id range: 1 -> %d", len(merged))
    logger.info("CSV header: %s", ",".join(CSV_FIELDNAMES))
    logger.info("Output: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
