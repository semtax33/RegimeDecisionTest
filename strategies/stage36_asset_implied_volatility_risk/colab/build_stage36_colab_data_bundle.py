from __future__ import annotations

import hashlib
import json
import unicodedata
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COLAB_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = COLAB_DIR / "stage36_colab_data.zip"


def normalized_path(directory: Path, filename: str) -> Path:
    target = unicodedata.normalize("NFC", filename)
    for path in directory.iterdir():
        if unicodedata.normalize("NFC", path.name) == target:
            return path
    raise FileNotFoundError(filename)


FILES = (
    (ROOT / "cache" / "market_daily.csv", "cache/market_daily.csv"),
    (
        ROOT / "cache" / "regime_lightgbm_ohlcv.csv",
        "cache/regime_lightgbm_ohlcv.csv",
    ),
    (ROOT / "raw_data" / "compass.db", "raw_data/compass.db"),
    (
        ROOT / "raw_data" / "krx_bond_index.csv",
        "raw_data/krx_bond_index.csv",
    ),
    (
        normalized_path(ROOT / "raw_data", "GDP 성장률.xlsx"),
        "raw_data/GDP 성장률.xlsx",
    ),
    (
        normalized_path(ROOT / "raw_data", "수출입 총괄_20260816.xlsx"),
        "raw_data/수출입 총괄_20260816.xlsx",
    ),
    (
        normalized_path(ROOT / "raw_data", "기업경기조사(전망).csv"),
        "raw_data/기업경기조사(전망).csv",
    ),
    (
        normalized_path(ROOT / "raw_data", "소비자물가 상승률.xlsx"),
        "raw_data/소비자물가 상승률.xlsx",
    ),
    (
        normalized_path(ROOT / "raw_data", "생산자물가 상승률.xlsx"),
        "raw_data/생산자물가 상승률.xlsx",
    ),
    (
        normalized_path(ROOT / "raw_data", "수출입물가 상승률.xlsx"),
        "raw_data/수출입물가 상승률.xlsx",
    ),
    (
        ROOT / "raw_data" / "VKOSPIData.csv",
        "raw_data/VKOSPIData.csv",
    ),
    (
        ROOT / "results" / "vix6_case1_features_daily.csv",
        "results/vix6_case1_features_daily.csv",
    ),
    (
        ROOT / "raw_data" / "260829_fwdPE.EPS.rev.xlsx",
        "raw_data/260829_fwdPE.EPS.rev.xlsx",
    ),
    (
        ROOT / "raw_data" / "260829_국고채.회사채.xlsx",
        "raw_data/260829_국고채.회사채.xlsx",
    ),
    (ROOT / "raw_data" / "GVZCLS.csv", "raw_data/GVZCLS.csv"),
    (ROOT / "raw_data" / "OVXCLS.csv", "raw_data/OVXCLS.csv"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle() -> dict:
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for source, archive_name in FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(
            {
                "path": archive_name,
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    manifest = {
        "bundle": "Stage36_GVZ_OVX_Colab_Data",
        "schema_version": 1,
        "root_directory": "stage36_data",
        "code_included": False,
        "files": records,
    }
    with zipfile.ZipFile(
        OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, archive_name in FILES:
            archive.write(source, f"stage36_data/{archive_name}")
        archive.writestr(
            "stage36_data/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    result = {
        **manifest,
        "output_zip": str(OUTPUT_ZIP.resolve()),
        "zip_bytes": OUTPUT_ZIP.stat().st_size,
        "zip_sha256": sha256(OUTPUT_ZIP),
    }
    (COLAB_DIR / "stage36_colab_data_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build_bundle(), ensure_ascii=False, indent=2))
