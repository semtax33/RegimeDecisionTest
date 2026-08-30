from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

SERIES = {
    "VIX": "VIXCLS",
    "OVX": "OVXCLS",
    "GVZ": "GVZCLS",
    "BAA_SPREAD": "BAA10Y",
    "NFCI": "NFCI",
    "STLFSI": "STLFSI4",
}

daily = []
for name, series_id in SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2005-01-01"
    frame = pd.read_csv(url)
    frame.columns = ["date", name]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    daily.append(frame.set_index("date"))

raw = pd.concat(daily, axis=1).sort_index()
raw.to_csv(CACHE / "stress_raw.csv")

monthly = pd.DataFrame(index=raw.resample("ME").last().index)
monthly["VIX_last"] = raw["VIX"].resample("ME").last()
monthly["VIX_mean"] = raw["VIX"].resample("ME").mean()
monthly["VIX_max"] = raw["VIX"].resample("ME").max()
monthly["OVX_last"] = raw["OVX"].resample("ME").last()
monthly["GVZ_last"] = raw["GVZ"].resample("ME").last()
monthly["BAA_SPREAD_last"] = raw["BAA_SPREAD"].resample("ME").last()
monthly["NFCI_last"] = raw["NFCI"].resample("ME").last()
monthly["STLFSI_last"] = raw["STLFSI"].resample("ME").last()
monthly.index = monthly.index.to_period("M")

for col in [
    "VIX_last", "VIX_mean", "VIX_max", "OVX_last", "GVZ_last",
    "BAA_SPREAD_last", "NFCI_last", "STLFSI_last",
]:
    monthly[f"{col}_d1"] = monthly[col].diff(1)
    monthly[f"{col}_d3"] = monthly[col].diff(3)
    rolling = monthly[col].rolling(60, min_periods=24)
    monthly[f"{col}_z60"] = (monthly[col] - rolling.mean()) / rolling.std(ddof=1)

monthly.to_csv(CACHE / "stress_monthly.csv")
print(monthly.loc["2007-01":].tail(12).round(3).to_string())
print("saved", CACHE / "stress_monthly.csv", "rows", len(monthly))
