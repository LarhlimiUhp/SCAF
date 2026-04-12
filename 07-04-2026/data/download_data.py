"""
download_data.py — Téléchargement des données de marché réelles via yfinance.

Exécuter UNE SEULE FOIS sur une machine avec accès internet :

    cd 07-04-2026
    python data/download_data.py

Les CSV sont sauvegardés dans data/csv/ et utilisés automatiquement par
MultiAssetLoader comme fallback quand yfinance est indisponible (sandbox CI,
Codespaces, etc.).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ─── configuration ────────────────────────────────────────────────────────── #

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

TICKERS = {
    "GSPC": "^GSPC",
    "VIX":  "^VIX",
    "TNX":  "^TNX",
    "IRX":  "^IRX",
    "GOLD": "GC=F",
    "OIL":  "CL=F",
}

OUT_DIR = Path(__file__).parent / "csv"

# ─── helpers ──────────────────────────────────────────────────────────────── #


def _download(yf, ticker: str, label: str) -> None:
    print(f"  Downloading {label} ({ticker}) …", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df is None or df.empty:
            print("EMPTY — skipped.")
            return

        # Flatten multi-index columns (newer yfinance versions)
        if hasattr(df.columns, "get_level_values"):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass

        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})

        keep = [c for c in ["Close", "Volume"] if c in df.columns]
        df = df[keep].dropna(subset=["Close"])

        out_path = OUT_DIR / f"{label}.csv"
        df.to_csv(out_path)
        print(f"OK ({len(df)} rows → {out_path.name})")
    except Exception as exc:
        print(f"FAILED ({exc})")


# ─── main ─────────────────────────────────────────────────────────────────── #


def main() -> None:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance is not installed. Run: pip install yfinance")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to: {OUT_DIR.resolve()}")
    print(f"Period: {START_DATE} → {END_DATE}\n")

    for label, ticker in TICKERS.items():
        _download(yf, ticker, label)

    print("\nDone. CSV files are ready for offline use.")
    print("Run the pipeline normally — MultiAssetLoader will use these files automatically.")


if __name__ == "__main__":
    main()
