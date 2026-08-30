from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from tools import download_krx_program_trading as krx_module
from tools.download_krx_program_trading import (
    CheckpointStore,
    DownloadResult,
    KrxClient,
    KrxResponseError,
    build_parser,
    iter_weekdays,
    parse_output_rows,
)


SAMPLE_PAYLOAD = {
    "output": [
        {
            "ITM_TP_NM": "차익",
            "ASK_TRDVOL": "1,234",
            "BID_TRDVOL": "2,345",
            "NETBID_TRDVOL": "1,111",
            "ASK_TRDVAL": "10,000",
            "BID_TRDVAL": "20,000",
            "NETBID_TRDVAL": "10,000",
        }
    ],
    "CURRENT_DATETIME": "2026.08.29 PM 02:44:47",
}


def make_result(payload=SAMPLE_PAYLOAD):
    return DownloadResult(
        payload=payload,
        raw_text=json.dumps(payload, ensure_ascii=False),
    )


def test_parse_output_rows_converts_formatted_numbers():
    rows = parse_output_rows(SAMPLE_PAYLOAD)
    assert rows[0]["category"] == "차익"
    assert rows[0]["ASK_TRDVOL"] == 1234
    assert rows[0]["NETBID_TRDVAL"] == 10000


def test_parse_output_rows_rejects_invalid_shape():
    with pytest.raises(KrxResponseError):
        parse_output_rows({"not_output": []})


def test_iter_weekdays_skips_saturday_and_sunday():
    assert list(iter_weekdays(date(2026, 8, 28), date(2026, 8, 31))) == [
        date(2026, 8, 28),
        date(2026, 8, 31),
    ]


def test_default_request_interval_is_two_seconds():
    assert build_parser().parse_args([]).delay == 2.0
    assert KrxClient("session=dummy").delay == 2.0


def test_request_interval_cannot_be_shorter_than_two_seconds():
    with pytest.raises(ValueError, match="2.0초 이상"):
        KrxClient("session=dummy", delay=1.99)


def test_rate_limiter_waits_until_two_seconds_have_elapsed(monkeypatch):
    client = KrxClient("session=dummy")
    client._last_request_started = 100.0
    sleeps = []
    monkeypatch.setattr(krx_module.time, "monotonic", lambda: 100.25)
    monkeypatch.setattr(krx_module.time, "sleep", sleeps.append)

    client._wait_for_rate_limit()

    assert sleeps == [pytest.approx(1.75)]


def test_checkpoint_is_resumable_and_exports_excel_friendly_csv(tmp_path):
    database = tmp_path / "checkpoint.sqlite3"
    output = tmp_path / "program.csv"
    store = CheckpointStore(database)
    try:
        trade_date = date(2026, 8, 28)
        assert not store.is_complete(trade_date, "ALL")
        assert store.save(trade_date, "ALL", make_result()) == 1
        assert store.is_complete(trade_date, "ALL")

        holiday = date(2026, 8, 27)
        assert store.save(holiday, "ALL", make_result({"output": []})) == 0
        assert store.is_complete(holiday, "ALL")

        assert store.export_csv(output, holiday, trade_date, "ALL") == 1
    finally:
        store.close()

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "date": "2026-08-28",
            "market": "ALL",
            "category": "차익",
            "sell_volume": "1234",
            "buy_volume": "2345",
            "net_buy_volume": "1111",
            "sell_value_krw": "10000",
            "buy_value_krw": "20000",
            "net_buy_value_krw": "10000",
        }
    ]


def test_empty_range_marks_each_covered_date_complete(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.sqlite3")
    try:
        start = date(2000, 1, 1)
        end = date(2000, 12, 31)
        store.save_empty_range(start, end, "ALL", make_result({"output": []}))
        assert store.is_complete(date(2000, 1, 3), "ALL")
        assert store.is_complete(date(2000, 12, 29), "ALL")
        assert not store.is_complete(date(2001, 1, 1), "ALL")
    finally:
        store.close()
