import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler


class CrossAssetFeatureEngineer:
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.feature_names = []
        self.scaler = StandardScaler()

    def build(self, spx_df, cross, cfg=None):
        if cross is None or not isinstance(cross, dict):
            raise TypeError('cross must be a dict of DataFrames')

        spx_close = spx_df['Close'].copy()
        spx_volume = spx_df.get('Volume', pd.Series(index=spx_close.index, dtype=float)).copy()
        spx_ret = np.log(spx_close / spx_close.shift(1))
        idx = spx_close.index
        feats = {}

        # === EXISTING FEATURES (ENHANCED) ===
        # Returns
        feats['ret_1d'] = spx_ret.shift(1)
        feats['ret_5d'] = spx_ret.rolling(5).sum().shift(1)
        feats['ret_10d'] = spx_ret.rolling(10).sum().shift(1)
        feats['ret_20d'] = spx_ret.rolling(20).sum().shift(1)

        # Volatility
        feats['vol_5d'] = spx_ret.rolling(5).std().shift(1)
        feats['vol_20d'] = spx_ret.rolling(20).std().shift(1)
        feats['vol_50d'] = spx_ret.rolling(50).std().shift(1)

        # Z-scores
        feats['zscore_20d'] = ((spx_close - spx_close.rolling(20).mean()) /
                               (spx_close.rolling(20).std() + 1e-6)).shift(1)
        feats['zscore_50d'] = ((spx_close - spx_close.rolling(50).mean()) /
                               (spx_close.rolling(50).std() + 1e-6)).shift(1)

        # RSI
        feats['rsi_7'] = self._rsi(spx_close, 7).shift(1)
        feats['rsi_14'] = self._rsi(spx_close, 14).shift(1)

        # Momentum
        feats['mom_10d'] = (spx_close / spx_close.shift(10) - 1).shift(1)
        feats['mom_20d'] = (spx_close / spx_close.shift(20) - 1).shift(1)

        # === MACRO-ECONOMIC FEATURES ===
        # Fed Rate Proxy (using 13-week treasury as proxy)
        irx = self._align(cross, 'irx', idx)
        if irx is not None:
            feats['fed_rate_proxy'] = irx.shift(1)
            feats['fed_rate_chg_5d'] = irx.pct_change(5).shift(1)

        # Yield Curve (10y - 3m spread)
        tnx = self._align(cross, 'tnx', idx)
        if tnx is not None and irx is not None:
            feats['yield_curve_spread'] = (tnx - irx).shift(1)
            feats['yield_curve_slope'] = ((tnx - irx) / (irx + 1e-6)).shift(1)
            feats['tnx_level'] = tnx.shift(1)
            feats['tnx_chg_5d'] = tnx.pct_change(5).shift(1)

        # Inflation Proxies (Gold and Oil as inflation hedges)
        gold = self._align(cross, 'gold', idx)
        oil = self._align(cross, 'oil', idx)
        if gold is not None:
            feats['gold_level'] = gold.shift(1)
            feats['gold_ret5'] = gold.pct_change(5).shift(1)
            feats['gold_vol20'] = gold.pct_change(1).rolling(20).std().shift(1)
        if oil is not None:
            feats['oil_level'] = oil.shift(1)
            feats['oil_ret5'] = oil.pct_change(5).shift(1)
            feats['oil_vol20'] = oil.pct_change(1).rolling(20).std().shift(1)

        # === SENTIMENT FEATURES ===
        # VIX (Fear Index)
        vix = self._align(cross, 'vix', idx)
        if vix is not None:
            feats['vix'] = vix.shift(1)
            feats['vix_d5'] = vix.pct_change(5).shift(1)
            feats['vix_d10'] = vix.pct_change(10).shift(1)
            feats['vix_zscore'] = ((vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-6)).shift(1)
            feats['vix_ma20'] = vix.rolling(20).mean().shift(1)

        # High Yield Spread (Credit Risk Sentiment)
        hyg = self._align(cross, 'hyg', idx)
        if hyg is not None and tnx is not None:
            hyg_ret = np.log(hyg / hyg.shift(1))
            feats['hyg_level'] = hyg.shift(1)
            feats['hyg_ret5'] = hyg_ret.rolling(5).sum().shift(1)
            feats['credit_spread_proxy'] = (hyg.pct_change(1) - tnx.pct_change(1)).shift(1)

        # === ADVANCED TECHNICAL FEATURES ===
        # MACD
        ema12 = spx_close.ewm(span=12).mean()
        ema26 = spx_close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        feats['macd'] = macd.shift(1)
        feats['macd_signal'] = signal.shift(1)
        feats['macd_hist'] = (macd - signal).shift(1)

        # Bollinger Bands
        sma20 = spx_close.rolling(20).mean()
        std20 = spx_close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        feats['bb_upper'] = bb_upper.shift(1)
        feats['bb_lower'] = bb_lower.shift(1)
        feats['bb_middle'] = sma20.shift(1)
        feats['bb_pct_b'] = ((spx_close - bb_lower) / (bb_upper - bb_lower + 1e-6)).shift(1)
        feats['bb_width'] = ((bb_upper - bb_lower) / sma20).shift(1)

        # Stochastic Oscillator
        high14 = spx_close.rolling(14).max()
        low14 = spx_close.rolling(14).min()
        stoch_k = 100 * ((spx_close - low14) / (high14 - low14 + 1e-6))
        stoch_d = stoch_k.rolling(3).mean()
        feats['stoch_k'] = stoch_k.shift(1)
        feats['stoch_d'] = stoch_d.shift(1)

        # Williams %R
        feats['williams_r'] = -100 * ((high14 - spx_close) / (high14 - low14 + 1e-6)).shift(1)

        # Rate of Change
        feats['roc_10'] = ((spx_close - spx_close.shift(10)) / spx_close.shift(10)).shift(1)
        feats['roc_20'] = ((spx_close - spx_close.shift(20)) / spx_close.shift(20)).shift(1)

        # === DERIVED FEATURES (INTERACTIONS & TRANSFORMATIONS) ===
        # Return-Volatility Interactions
        if 'ret_1d' in feats and 'vol_20d' in feats:
            feats['ret_vol_ratio'] = (feats['ret_1d'] / (feats['vol_20d'] + 1e-6))
            feats['ret_vol_product'] = feats['ret_1d'] * feats['vol_20d']

        # Momentum-Volatility
        if 'mom_10d' in feats and 'vol_20d' in feats:
            feats['mom_vol_ratio'] = feats['mom_10d'] / (feats['vol_20d'] + 1e-6)

        # Transformations
        feats['ret_1d_log'] = np.log(feats['ret_1d'] + 2)  # Shift to avoid log(0)
        feats['vol_20d_sqrt'] = np.sqrt(feats['vol_20d'] + 1e-6)

        # === SECTORIAL FEATURES ===
        sectors = ['xlk', 'xlv', 'rsp', 'qqq', 'eem']
        for sector in sectors:
            sector_data = self._align(cross, sector, idx)
            if sector_data is not None:
                sector_ret = np.log(sector_data / sector_data.shift(1))
                feats[f'{sector}_ret5'] = sector_ret.rolling(5).sum().shift(1)
                feats[f'{sector}_level'] = sector_data.shift(1)
                feats[f'{sector}_vol20'] = sector_ret.rolling(20).std().shift(1)
                feats[f'{sector}_mom10'] = (sector_data / sector_data.shift(10) - 1).shift(1)

        # Sector vs SPX spreads
        spy = self._align(cross, 'spy', idx)
        if spy is not None:
            spy_ret = np.log(spy / spy.shift(1))
            feats['spy_ret5'] = spy_ret.rolling(5).sum().shift(1)
            for sector in sectors:
                if f'{sector}_ret5' in feats:
                    feats[f'{sector}_spy_spread'] = feats[f'{sector}_ret5'] - feats['spy_ret5']

        # === HIGH-FREQUENCY STYLE FEATURES (DAILY DATA PROXIES) ===
        # Volume-based features
        if not spx_volume.empty and not spx_volume.isna().all():
            vol_ma20 = spx_volume.rolling(20).mean()
            feats['volume_ratio'] = (spx_volume / (vol_ma20 + 1e-6)).shift(1)
            feats['volume_zscore'] = ((spx_volume - vol_ma20) / (spx_volume.rolling(20).std() + 1e-6)).shift(1)

        # Gap features (open vs close, but we don't have open, using prev close)
        prev_close = spx_close.shift(1)
        feats['gap_up'] = ((spx_close - prev_close) > 0.01).astype(float).shift(1)  # 1% gap
        feats['gap_down'] = ((spx_close - prev_close) < -0.01).astype(float).shift(1)

        # Intraday volatility proxy (using range if available, else ret magnitude)
        feats['daily_range_proxy'] = np.abs(spx_ret).shift(1)

        # === DXY FEATURES ===
        dxy = self._align(cross, 'dxy', idx)
        if dxy is not None:
            feats['dxy_level'] = dxy.shift(1)
            feats['dxy_r5'] = dxy.pct_change(5).shift(1)
            feats['dxy_r10'] = dxy.pct_change(10).shift(1)
            feats['dxy_zscore'] = ((dxy - dxy.rolling(20).mean()) / (dxy.rolling(20).std() + 1e-6)).shift(1)

        # Cross-Asset Ratios (after dxy is defined)
        if vix is not None and 'dxy_level' in feats:
            feats['vix_dxy_ratio'] = (vix / feats['dxy_level']).shift(1)

        # === ADDITIONAL ADVANCED FEATURES ===

        # Ichimoku Cloud components
        tenkan_sen = (spx_close.rolling(9).max() + spx_close.rolling(9).min()) / 2
        kijun_sen = (spx_close.rolling(26).max() + spx_close.rolling(26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((spx_close.rolling(52).max() + spx_close.rolling(52).min()) / 2).shift(26)
        chikou_span = spx_close.shift(-26)

        feats['ichimoku_tenkan'] = tenkan_sen.shift(1)
        feats['ichimoku_kijun'] = kijun_sen.shift(1)
        feats['ichimoku_senkou_a'] = senkou_span_a.shift(1)
        feats['ichimoku_senkou_b'] = senkou_span_b.shift(1)
        # chikou_span uses a negative shift; clamp to avoid forward-looking NaN
        feats['ichimoku_chikou'] = chikou_span.shift(1).ffill()

        # Commodity Channel Index (CCI)
        typical_price = (spx_close + spx_close.rolling(20).max() + spx_close.rolling(20).min()) / 3
        sma_tp = typical_price.rolling(20).mean()
        mean_dev = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        feats['cci'] = ((typical_price - sma_tp) / (0.015 * mean_dev)).shift(1)

        # Ultimate Oscillator
        bp = spx_close - np.minimum(spx_close.shift(1), spx_close.rolling(14).min())
        tr = np.maximum(spx_close - spx_close.shift(1),
                       np.maximum(np.abs(spx_close - spx_close.shift(1)),
                                np.abs(spx_close - spx_close.shift(1))))
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        feats['ultimate_oscillator'] = (4 * avg7 + 2 * avg14 + avg28) / 7 * 100
        feats['ultimate_oscillator'] = feats['ultimate_oscillator'].shift(1)

        # Chaikin Money Flow
        mfv = ((spx_close - spx_close.shift(1)) / spx_close.shift(1)) * spx_volume
        feats['cmf'] = (mfv.rolling(21).sum() / spx_volume.rolling(21).sum()).shift(1)

        # Volume Weighted Average Price (VWAP)
        vwap = (spx_volume * (spx_close + spx_close.shift(1) + spx_close.shift(1)) / 3).cumsum() / spx_volume.cumsum()
        feats['vwap'] = vwap.shift(1)
        feats['vwap_ratio'] = (spx_close / vwap).shift(1)

        # Clean and prepare data
        df = pd.DataFrame(feats, index=idx)
        # Replace infinities, forward-fill isolated NaNs produced by rolling windows,
        # then backward-fill any leading NaNs, and finally zero-fill any remainder.
        df = (df
              .replace([np.inf, -np.inf], np.nan)
              .ffill()
              .bfill()
              .fillna(0))

        # Target
        fwd_ret = spx_ret.rolling(self.horizon).sum().shift(-self.horizon)
        target = (fwd_ret > 0).astype(float)

        # Align all data
        valid = target.notna() & df.notna().all(axis=1)
        df = df.loc[valid]
        target = target.loc[valid]
        spx_close = spx_close.loc[valid]
        spx_ret_v = spx_ret.loc[valid]

        self.feature_names = list(df.columns)
        print(f'Enriched feature set: {len(self.feature_names)} features')
        print(f'Features: {self.feature_names[:10]}...')  # Show first 10

        if cfg is not None:
            df = self._filter_features(df, target, cfg)

        self.feature_names = list(df.columns)
        print(f'Selected top feature set: {len(self.feature_names)} features')
        print(f'Selected features: {self.feature_names[:10]}...')

        return df, target, spx_close, spx_ret_v

    def _filter_features(self, df, target, cfg):
        if cfg is None or cfg.N_TOP_FEATURES >= df.shape[1]:
            return df

        # Étape 1: Réduction des features fortement corrélées (seuil plus strict)
        df = self._reduce_correlated_features(df, threshold=0.85)  # Réduit de 0.95 à 0.85

        if df.shape[1] <= cfg.N_TOP_FEATURES:
            return df

        # Étape 2: Sélection basée sur l'importance avec LightGBM
        try:
            import lightgbm as lgb

            # Utiliser LightGBM pour évaluer l'importance
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )

            # Calculer l'importance des features
            lgb_model.fit(df.values, target.values)
            importance_scores = lgb_model.feature_importances_

            # Sélectionner les top features
            feature_importance = pd.DataFrame({
                'feature': df.columns,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)

            # Prendre les N_TOP_FEATURES plus importantes
            selected_features = feature_importance.head(cfg.N_TOP_FEATURES)['feature'].tolist()
            return df[selected_features]

        except Exception as e:
            print(f"Erreur lors de la sélection LightGBM: {e}, utilisation de SelectKBest")
            # Fallback vers SelectKBest
            selector = SelectKBest(score_func=mutual_info_classif, k=min(cfg.N_TOP_FEATURES, df.shape[1]))
            selector.fit(df.fillna(0).values, target.values)
            mask = selector.get_support()
            selected_columns = df.columns[mask]
            return df[selected_columns]

    @staticmethod
    def _reduce_correlated_features(df, threshold=0.95):
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)

    @staticmethod
    def _align(cross, name, idx):
        df = cross.get(name)
        if df is None:
            return None
        return df['Close'].reindex(idx).ffill()

    @staticmethod
    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-6))
