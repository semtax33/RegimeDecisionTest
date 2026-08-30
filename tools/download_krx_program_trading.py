"""Download daily KRX program-trading totals with durable resume support.

The KRX endpoint aggregates the requested date range.  To produce a genuinely
daily data set, this module sends one request for every weekday and attaches the
requested date to the returned rows.  Each successful response (including an
empty holiday response) is committed to SQLite before the next request starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENDPOINT = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
BUILD_ID = "dbms/MDC/STAT/standard/MDCSTAT02601"
MIN_REQUEST_INTERVAL_SECONDS = 2.0
REFERER = (
    "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/"
    "index.cmd?menuId=MDC0201020305"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0.0.0 Safari/537.36"
)
NUMERIC_FIELDS = (
    "ASK_TRDVOL",
    "BID_TRDVOL",
    "NETBID_TRDVOL",
    "ASK_TRDVAL",
    "BID_TRDVAL",
    "NETBID_TRDVAL",
)
CSV_HEADER = (
    "date",
    "market",
    "category",
    "sell_volume",
    "buy_volume",
    "net_buy_volume",
    "sell_value_krw",
    "buy_value_krw",
    "net_buy_value_krw",
)


class KrxDownloadError(RuntimeError):
    """Base class for KRX download failures."""


class KrxAuthenticationError(KrxDownloadError):
    """Raised when the supplied KRX browser session is no longer valid."""


class KrxResponseError(KrxDownloadError):
    """Raised when KRX returns an unexpected response."""


@dataclass(frozen=True)
class DownloadResult:
    payload: dict[str, Any]
    raw_text: str


def parse_yyyymmdd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"날짜는 YYYYMMDD 형식이어야 합니다: {value!r}"
        ) from exc


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_weekdays(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def parse_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise KrxResponseError(f"{field_name} 값이 숫자가 아닙니다: {value!r}")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise KrxResponseError(f"{field_name} 값이 문자열/정수가 아닙니다: {value!r}")
    normalized = value.replace(",", "").strip()
    try:
        return int(normalized)
    except ValueError as exc:
        raise KrxResponseError(
            f"{field_name} 값을 정수로 변환할 수 없습니다: {value!r}"
        ) from exc


def parse_output_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = payload.get("output")
    if not isinstance(output, list):
        raise KrxResponseError("응답 JSON에 output 배열이 없습니다.")

    parsed: list[dict[str, Any]] = []
    for row in output:
        if not isinstance(row, dict) or not isinstance(row.get("ITM_TP_NM"), str):
            raise KrxResponseError(f"예상하지 못한 output 행입니다: {row!r}")
        parsed.append(
            {
                "category": row["ITM_TP_NM"].strip(),
                **{field: parse_integer(row.get(field), field) for field in NUMERIC_FIELDS},
            }
        )
    return parsed


class KrxClient:
    def __init__(
        self,
        cookie: str,
        *,
        delay: float = MIN_REQUEST_INTERVAL_SECONDS,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        if not cookie.strip():
            raise ValueError("KRX 쿠키가 비어 있습니다.")
        if delay < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError(
                f"KRX 차단 방지를 위해 요청 간격은 "
                f"{MIN_REQUEST_INTERVAL_SECONDS:.1f}초 이상이어야 합니다."
            )
        if max_retries < 1:
            raise ValueError("최대 재시도 횟수는 1 이상이어야 합니다.")
        self.cookie = cookie.strip()
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_started: float | None = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_started is None:
            return
        remaining = self.delay - (time.monotonic() - self._last_request_started)
        if remaining > 0:
            time.sleep(remaining)

    def _request_once(
        self, start_date: date, market: str, end_date: date | None = None
    ) -> DownloadResult:
        self._wait_for_rate_limit()
        end_date = end_date or start_date
        body = urlencode(
            {
                "bld": BUILD_ID,
                "locale": "ko_KR",
                "mktId": market,
                "strtDd": start_date.strftime("%Y%m%d"),
                "endDd": end_date.strftime("%Y%m%d"),
                # Ask for base units so that the normalized database contains
                # shares and KRW, with no display-unit ambiguity.
                "share": "1",
                "money": "1",
                "csvxls_isNo": "false",
            }
        ).encode("ascii")
        request = Request(
            ENDPOINT,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Cookie": self.cookie,
                "Origin": "https://data.krx.co.kr",
                "Referer": REFERER,
                "User-Agent": USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        self._last_request_started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_text = response.read().decode("utf-8-sig")
        except HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            if exc.code in {400, 401, 403} and (
                "LOGOUT" in error_text.upper() or exc.code in {401, 403}
            ):
                raise KrxAuthenticationError(
                    "KRX 로그인 세션이 만료되었거나 쿠키가 유효하지 않습니다. "
                    "브라우저에서 새 Cookie 헤더를 복사한 뒤 같은 명령을 다시 실행하세요."
                ) from exc
            raise KrxDownloadError(
                f"KRX HTTP {exc.code}: {error_text[:300]!r}"
            ) from exc
        except URLError as exc:
            raise KrxDownloadError(f"KRX 연결 실패: {exc.reason}") from exc

        if raw_text.strip().upper() == "LOGOUT":
            raise KrxAuthenticationError("KRX 로그인 세션이 만료되었습니다.")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            if "로그인" in raw_text or "LOGOUT" in raw_text.upper():
                raise KrxAuthenticationError("KRX 로그인 세션이 만료되었습니다.") from exc
            raise KrxResponseError(
                f"KRX 응답이 JSON이 아닙니다: {raw_text[:300]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise KrxResponseError("KRX 응답의 최상위 값이 객체가 아닙니다.")
        parse_output_rows(payload)
        return DownloadResult(payload=payload, raw_text=raw_text)

    def fetch(
        self, start_date: date, market: str, end_date: date | None = None
    ) -> DownloadResult:
        label = (
            start_date.isoformat()
            if end_date is None or end_date == start_date
            else f"{start_date.isoformat()}~{end_date.isoformat()}"
        )
        last_error: KrxDownloadError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._request_once(start_date, market, end_date)
            except KrxAuthenticationError:
                raise
            except KrxResponseError:
                # A structurally invalid 200 response should not be hammered.
                raise
            except KrxDownloadError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                backoff = min(60.0, 2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(
                    f"[{label}] 요청 실패 ({attempt}/{self.max_retries}): "
                    f"{exc}; {backoff:.1f}초 후 재시도",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(backoff)
        assert last_error is not None
        raise last_error


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_fetch (
                trade_date TEXT NOT NULL,
                market TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('ok', 'empty')),
                row_count INTEGER NOT NULL,
                response_json TEXT NOT NULL,
                server_datetime TEXT,
                fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY (trade_date, market)
            );

            CREATE TABLE IF NOT EXISTS program_trading (
                trade_date TEXT NOT NULL,
                market TEXT NOT NULL,
                category TEXT NOT NULL,
                sell_volume INTEGER NOT NULL,
                buy_volume INTEGER NOT NULL,
                net_buy_volume INTEGER NOT NULL,
                sell_value_krw INTEGER NOT NULL,
                buy_value_krw INTEGER NOT NULL,
                net_buy_value_krw INTEGER NOT NULL,
                PRIMARY KEY (trade_date, market, category),
                FOREIGN KEY (trade_date, market)
                    REFERENCES daily_fetch (trade_date, market)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_program_trading_date
                ON program_trading (trade_date, market);

            CREATE TABLE IF NOT EXISTS empty_ranges (
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                market TEXT NOT NULL,
                response_json TEXT NOT NULL,
                server_datetime TEXT,
                fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY (start_date, end_date, market)
            );

            CREATE INDEX IF NOT EXISTS idx_empty_ranges_coverage
                ON empty_ranges (market, start_date, end_date);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def is_complete(self, trade_date: date, market: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            WHERE EXISTS (
                SELECT 1 FROM daily_fetch
                WHERE trade_date = ? AND market = ?
            ) OR EXISTS (
                SELECT 1 FROM empty_ranges
                WHERE market = ? AND start_date <= ? AND end_date >= ?
            )
            """,
            (
                trade_date.isoformat(),
                market,
                market,
                trade_date.isoformat(),
                trade_date.isoformat(),
            ),
        ).fetchone()
        return row is not None

    def has_data_in_range(self, start: date, end: date, market: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM program_trading
            WHERE trade_date BETWEEN ? AND ? AND market = ?
            LIMIT 1
            """,
            (start.isoformat(), end.isoformat(), market),
        ).fetchone()
        return row is not None

    def save_empty_range(
        self,
        start: date,
        end: date,
        market: str,
        result: DownloadResult,
    ) -> None:
        if parse_output_rows(result.payload):
            raise ValueError("데이터가 있는 응답은 빈 범위로 저장할 수 없습니다.")
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO empty_ranges (
                    start_date, end_date, market, response_json,
                    server_datetime, fetched_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (start_date, end_date, market) DO UPDATE SET
                    response_json = excluded.response_json,
                    server_datetime = excluded.server_datetime,
                    fetched_at_utc = excluded.fetched_at_utc
                """,
                (
                    start.isoformat(),
                    end.isoformat(),
                    market,
                    json.dumps(result.payload, ensure_ascii=False, separators=(",", ":")),
                    result.payload.get("CURRENT_DATETIME"),
                    iso_utc_now(),
                ),
            )

    def save(self, trade_date: date, market: str, result: DownloadResult) -> int:
        rows = parse_output_rows(result.payload)
        key = (trade_date.isoformat(), market)
        with self.connection:
            self.connection.execute(
                "DELETE FROM program_trading WHERE trade_date = ? AND market = ?",
                key,
            )
            self.connection.execute(
                """
                INSERT INTO daily_fetch (
                    trade_date, market, state, row_count, response_json,
                    server_datetime, fetched_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (trade_date, market) DO UPDATE SET
                    state = excluded.state,
                    row_count = excluded.row_count,
                    response_json = excluded.response_json,
                    server_datetime = excluded.server_datetime,
                    fetched_at_utc = excluded.fetched_at_utc
                """,
                (
                    *key,
                    "ok" if rows else "empty",
                    len(rows),
                    json.dumps(result.payload, ensure_ascii=False, separators=(",", ":")),
                    result.payload.get("CURRENT_DATETIME"),
                    iso_utc_now(),
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO program_trading (
                    trade_date, market, category,
                    sell_volume, buy_volume, net_buy_volume,
                    sell_value_krw, buy_value_krw, net_buy_value_krw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        *key,
                        row["category"],
                        row["ASK_TRDVOL"],
                        row["BID_TRDVOL"],
                        row["NETBID_TRDVOL"],
                        row["ASK_TRDVAL"],
                        row["BID_TRDVAL"],
                        row["NETBID_TRDVAL"],
                    )
                    for row in rows
                ],
            )
        return len(rows)

    def export_csv(self, output_path: Path, start: date, end: date, market: str) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        query = self.connection.execute(
            """
            SELECT
                trade_date, market, category,
                sell_volume, buy_volume, net_buy_volume,
                sell_value_krw, buy_value_krw, net_buy_value_krw
            FROM program_trading
            WHERE trade_date BETWEEN ? AND ? AND market = ?
            ORDER BY
                trade_date,
                CASE category
                    WHEN '차익' THEN 1
                    WHEN '비차익' THEN 2
                    WHEN '전체' THEN 3
                    ELSE 4
                END,
                category
            """,
            (start.isoformat(), end.isoformat(), market),
        )
        row_count = 0
        try:
            with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(CSV_HEADER)
                for row in query:
                    writer.writerow(row)
                    row_count += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return row_count


def read_cookie(args: argparse.Namespace) -> str:
    if args.cookie:
        return args.cookie.strip()
    if args.cookie_file:
        return args.cookie_file.read_text(encoding="utf-8-sig").strip()
    return os.environ.get("KRX_COOKIE", "").strip()


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "KRX 프로그램매매 자료를 일별로 받아 SQLite 체크포인트와 CSV로 저장합니다."
        )
    )
    parser.add_argument("--start-date", type=parse_yyyymmdd, default=parse_yyyymmdd("20000101"))
    parser.add_argument(
        "--end-date",
        type=parse_yyyymmdd,
        default=date.today(),
        help="종료일 YYYYMMDD (기본값: 오늘)",
    )
    parser.add_argument("--market", choices=("ALL", "STK", "KSQ"), default="ALL")
    parser.add_argument(
        "--delay",
        type=float,
        default=MIN_REQUEST_INTERVAL_SECONDS,
        help=(
            f"요청 시작 간 최소 간격(초, 기본/최솟값: "
            f"{MIN_REQUEST_INTERVAL_SECONDS:.1f})"
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--database",
        type=Path,
        default=project_root / "raw_data" / "krx_program_trading.sqlite3",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "raw_data" / "krx_program_trading.csv",
    )
    cookie_group = parser.add_mutually_exclusive_group()
    cookie_group.add_argument(
        "--cookie",
        help="브라우저의 Cookie 요청 헤더 (명령 기록 노출 방지를 위해 환경변수 권장)",
    )
    cookie_group.add_argument(
        "--cookie-file",
        type=Path,
        help="Cookie 요청 헤더 한 줄이 든 UTF-8 파일",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="네트워크 요청 없이 현재 SQLite 내용을 CSV로 다시 생성",
    )
    parser.add_argument(
        "--no-year-probe",
        action="store_true",
        help=(
            "연도 전체가 빈 구간인지 먼저 확인하는 최적화를 끕니다. "
            "기본적으로 빈 연도는 요청 1회로 검증하고 일별 요청을 생략합니다."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("시작일이 종료일보다 늦습니다.")
    if args.progress_every < 1:
        raise ValueError("--progress-every는 1 이상이어야 합니다.")

    weekdays = list(iter_weekdays(args.start_date, args.end_date))
    store = CheckpointStore(args.database)
    try:
        if args.export_only:
            row_count = store.export_csv(
                args.csv, args.start_date, args.end_date, args.market
            )
            print(f"CSV 생성 완료: {args.csv} ({row_count:,}행)", flush=True)
            return 0

        cookie = read_cookie(args)
        if not cookie:
            raise ValueError(
                "KRX 쿠키가 필요합니다. PowerShell에서 $env:KRX_COOKIE를 설정하거나 "
                "--cookie-file을 지정하세요."
            )
        client = KrxClient(
            cookie,
            delay=args.delay,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        pending = [day for day in weekdays if not store.is_complete(day, args.market)]
        already_complete = len(weekdays) - len(pending)
        print(
            f"범위={args.start_date}~{args.end_date} 시장={args.market} "
            f"평일={len(weekdays):,} 완료={already_complete:,} "
            f"남음={len(pending):,} 요청간격={args.delay:.3f}초",
            flush=True,
        )

        if not args.no_year_probe:
            first_year = args.start_date.year
            last_year = args.end_date.year
            for year in range(first_year, last_year + 1):
                range_start = max(args.start_date, date(year, 1, 1))
                range_end = min(args.end_date, date(year, 12, 31))
                range_weekdays = list(iter_weekdays(range_start, range_end))
                if not range_weekdays:
                    continue
                if all(store.is_complete(day, args.market) for day in range_weekdays):
                    continue
                # A previously saved non-empty day proves that this year cannot
                # be skipped, so avoid spending another request on the probe.
                if store.has_data_in_range(range_start, range_end, args.market):
                    continue
                print(
                    f"빈 연도 확인: {range_start}~{range_end}",
                    flush=True,
                )
                probe_result = client.fetch(range_start, args.market, range_end)
                if not parse_output_rows(probe_result.payload):
                    store.save_empty_range(
                        range_start, range_end, args.market, probe_result
                    )
                    print(
                        f"빈 연도 확정: {year} ({len(range_weekdays):,}개 평일 생략)",
                        flush=True,
                    )

            pending = [day for day in weekdays if not store.is_complete(day, args.market)]
            already_complete = len(weekdays) - len(pending)
            print(
                f"연도 확인 후 완료={already_complete:,} 남음={len(pending):,}",
                flush=True,
            )

        fetched_this_run = 0
        nonempty_dates = 0
        empty_dates = 0
        started_at = time.monotonic()
        for trade_date in pending:
            result = client.fetch(trade_date, args.market)
            row_count = store.save(trade_date, args.market, result)
            fetched_this_run += 1
            if row_count:
                nonempty_dates += 1
            else:
                empty_dates += 1

            if fetched_this_run % args.progress_every == 0 or fetched_this_run == len(pending):
                elapsed = time.monotonic() - started_at
                rate = fetched_this_run / elapsed if elapsed else 0.0
                remaining = len(pending) - fetched_this_run
                eta_minutes = remaining / rate / 60.0 if rate else 0.0
                print(
                    f"진행 {fetched_this_run:,}/{len(pending):,} "
                    f"({trade_date}, 데이터={nonempty_dates:,}, 빈날짜={empty_dates:,}, "
                    f"{rate:.2f}요청/초, 예상잔여={eta_minutes:.1f}분)",
                    flush=True,
                )

        csv_rows = store.export_csv(args.csv, args.start_date, args.end_date, args.market)
        print(
            f"완료: DB={args.database} CSV={args.csv} CSV행={csv_rows:,}",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        csv_rows = store.export_csv(args.csv, args.start_date, args.end_date, args.market)
        print(
            f"\n중단됨: 현재까지 {csv_rows:,}행을 CSV로 내보냈습니다. "
            "같은 명령을 재실행하면 이어받습니다.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except KrxDownloadError as exc:
        csv_rows = store.export_csv(args.csv, args.start_date, args.end_date, args.market)
        print(
            f"다운로드 중단: {exc}\n현재까지 {csv_rows:,}행을 CSV로 내보냈습니다. "
            "원인을 해결하고 같은 명령을 재실행하면 이어받습니다.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    finally:
        store.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
