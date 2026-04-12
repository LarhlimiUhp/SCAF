from pathlib import Path

import pandas as pd

# Directory where pre-downloaded CSV files are stored (populated by data/download_data.py)
_CSV_DIR = Path(__file__).parent / "csv"

# Mapping: cross-asset config key → CSV filename stem (without .csv)
_CROSS_CSV_MAP = {
    "vix":  "VIX",
    "tnx":  "TNX",
    "irx":  "IRX",
    "gold": "GOLD",
    "oil":  "OIL",
}

# Main ticker CSV filename stem
_MAIN_CSV_STEM = "GSPC"


class MultiAssetLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def download(self):
        if not self.cfg.USE_REAL_DATA:
            print('Using synthetic data as configured')
            return self._load_synthetic_data()

        start = self.cfg.START_DATE
        end = self.cfg.END_DATE or self.cfg.end_date()

        # ── (1) Try pre-downloaded CSV files (works offline / sandbox) ── #
        spx = self._load_csv(_MAIN_CSV_STEM, start, end)
        if spx is not None and len(spx) >= 500:
            print(f'  Loaded {_MAIN_CSV_STEM}.csv from local cache ({len(spx)} rows)')
            cross = {}
            for name in self.cfg.CROSS_ASSET_TICKERS:
                stem = _CROSS_CSV_MAP.get(name.lower())
                if stem:
                    df = self._load_csv(stem, start, end)
                    if df is not None and len(df) > 100:
                        cross[name] = df
                        print(f'    cross asset {name}: {len(df)} rows (CSV)')
                    else:
                        print(f'    Warning: CSV unavailable or insufficient for {name}')
            if len(cross) == 0:
                print('Warning: No cross-asset series loaded from CSV. Proceeding with S&P 500 only.')
            return spx, cross

        # ── (2) Try live yfinance download ─────────────────────────────── #
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                'USE_REAL_DATA is enabled but yfinance is not installed. '
                'Install yfinance to download real market data, or run '
                'data/download_data.py once to create local CSV files.'
            ) from exc

        spx = self._download_series(yf, self.cfg.TICKER, start, end, 'S&P 500')
        if spx is None or len(spx) < 500:
            raise RuntimeError(
                f'USE_REAL_DATA is enabled but failed to download sufficient real data for '
                f'{self.cfg.TICKER} ({len(spx) if spx is not None else 0} rows). '
                f'Run data/download_data.py once to create local CSV files for offline use.'
            )

        cross = {}
        for name, ticker in self.cfg.CROSS_ASSET_TICKERS.items():
            df = self._download_series(yf, ticker, start, end, name)
            if df is not None and len(df) > 100:
                cross[name] = df
            else:
                print(f'  Warning: real data unavailable or insufficient for cross asset {name} ({ticker})')

        if len(cross) == 0:
            print('Warning: No cross-asset series loaded from real data. Proceeding with S&P 500 only.')

        return spx, cross

    # ── CSV helpers ───────────────────────────────────────────────────────── #

    @staticmethod
    def _load_csv(stem: str, start: str, end: str) -> "pd.DataFrame | None":
        """Load a pre-downloaded CSV from data/csv/<stem>.csv and filter by date range."""
        csv_path = _CSV_DIR / f"{stem}.csv"
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Filter by date range
            df = df.loc[start:end]
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            if "Close" not in df.columns:
                return None
            df = df[[c for c in ["Close", "Volume"] if c in df.columns]]
            df = df.dropna(subset=["Close"])
            return df if len(df) > 0 else None
        except Exception:
            return None

    def _load_synthetic_data(self):
        """Load synthetic data for testing when real data is unavailable"""
        try:
            from create_test_data import create_synthetic_data
            print('Loading synthetic market data...')
            return create_synthetic_data()
        except ImportError:
            print('Synthetic data generator not found')
            return None, None

    @staticmethod
    def _download_series(yf, ticker, start, end, label):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df is None or df.empty:
                return None

            # Handle multi-index columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-index columns
                df.columns = df.columns.get_level_values(0)

            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})
            if 'Close' not in df.columns:
                return None
            df = df[[c for c in ['Close', 'Volume'] if c in df.columns]]
            df = df.dropna(subset=['Close'])
            return df
        except Exception:
            return None
