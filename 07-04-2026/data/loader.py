import pandas as pd


class MultiAssetLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def download(self):
        if not self.cfg.USE_REAL_DATA:
            print('Using synthetic data as configured')
            return self._load_synthetic_data()

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                'USE_REAL_DATA is enabled but yfinance is not installed. '
                'Install yfinance to download real market data.'
            ) from exc

        start = self.cfg.START_DATE
        end = self.cfg.END_DATE or self.cfg.end_date()
        spx = self._download_series(yf, self.cfg.TICKER, start, end, 'S&P 500')
        if spx is None or len(spx) < 500:
            raise RuntimeError(
                f'USE_REAL_DATA is enabled but failed to download sufficient real data for '
                f'{self.cfg.TICKER} ({len(spx) if spx is not None else 0} rows).'
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
