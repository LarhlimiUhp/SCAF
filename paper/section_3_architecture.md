# Section 3 — Architecture SCAF

## 3.1 Vue d'ensemble

SCAF est une architecture hybride à trois couches fonctionnelles :
(i) **extraction de signal**, (ii) **fusion de signal**, et (iii) **gestion du risque**.

```
┌───────────────────────────────────────────────────────────────────┐
│                    DONNÉES D'ENTRÉE                               │
│   Prix OHLCV (18 ETFs, quotidien)  +  Indices sentiments (6)     │
└───────────────────────┬───────────────────────────────────────────┘
                        │
          ┌─────────────┼──────────────────┐
          ▼             ▼                  ▼
  ┌───────────────┐ ┌──────────────┐  ┌──────────────────────────┐
  │  TrendEngine  │ │  Features    │  │  Features cross-section  │
  │  (Donchian +  │ │  Panel       │  │  (58 features / actif)   │
  │  TF blended)  │ │  (§3.2)      │  │                          │
  └───────┬───────┘ └──────┬───────┘  └────────────┬─────────────┘
          │                │                        │
          │ trend_signal   │ features_dict          │
          │ ∈ [-1, 1]      │ (dates × tickers × F)  │
          │                ▼                        ▼
          │         ┌──────────────────────────────────────────────┐
          │         │        TCCA (Section 4)                      │
          │         │  Q = f(trend_signal)                         │
          │         │  K, V = g(sector features)                   │
          │         │  scores = Attention(Q, K, V)                 │
          │         └───────────────────┬──────────────────────────┘
          │                             │ scores ∈ ℝ^18
          │                             ▼
          │                  CrossSectionPortfolio
          │                  (sigmoid → long/short weights)
          │                             │ cs_weights ∈ ℝ^18
          ▼                             ▼
  ┌───────────────────────────────────────────────────────────────┐
  │            HybridSignal (§3.4)                                │
  │    pnl_raw = w_trend × trend_pnl + w_cs × cs_pnl             │
  │    w_trend ≈ 0.99 (dominant)  |  w_cs ≈ 0.01–0.05           │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │            RiskManager (§3.5)                                 │
  │    VolTargeting + DrawdownTiers → pnl_adj                     │
  └───────────────────────────────────────────────────────────────┘
```

## 3.2 Univers et features

### 3.2.1 Univers d'actifs

L'univers est composé de 18 ETFs sectoriels américains du SPDR Select Sector
(XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, GLD, SLV, TLT,
IEF, HYG, EEM, SPY) sur la période 2015–2024. SPY joue le rôle de benchmark
pour le signal de tendance et l'évaluation des baselines.
Les données sont téléchargées via yfinance et mises en cache localement.
Deux actifs (XLC, XLRE) sont exclus dynamiquement pour données manquantes
excessives (NaN > 5% sur Close), ramenant l'univers effectif à 18 actifs.

### 3.2.2 Feature engineering

Pour chaque actif, un panel de 58 features est calculé sur la fenêtre glissante
quotidienne, organisé en quatre familles :

**Features de momentum** (12 features) : rendements log sur horizons 1, 3, 5,
10, 20, 60 jours ; RSI(14) ; MACD et signal MACD.

**Features de volatilité** (8 features) : volatilité réalisée sur 5, 10, 20,
60 jours ; Average True Range (ATR) ; ratio volatilité court/long terme.

**Features de tendance** (10 features) : signaux Donchian (position relative
prix/canal sur fenêtres 10, 20, 50) ; écarts aux moyennes mobiles exponentielles.

**Features cross-sectionnelles** (16 features) : rangs relatifs de chaque actif
sur toutes les dimensions ci-dessus, calculés par rapport à l'ensemble de
l'univers à chaque date. Ces features cross-sectionnelles capturent la position
relative de l'actif dans l'univers et constituent le signal d'entrée principal
du TCCA.

**Features de sentiment** (6 features) : indices de sentiment agrégés (VIX,
put/call ratio, bull-bear spread, AAII, II Survey, CNN Fear & Greed).

Tous les features sont normalisés par rolling z-score (fenêtre 252 jours) pour
assurer la stationnarité des distributions d'entrée.

### 3.2.3 Cible d'entraînement

La cible d'entraînement du TCCA est le rendement log forward à 5 jours :

```
y_t^(i) = log(P_{t+5}^(i) / P_t^(i))
```

où P_t^(i) est le prix de clôture de l'actif i à la date t. L'objectif
d'apprentissage est de ranker les actifs par rendement futur anticipé,
non de prédire la valeur absolue de ce rendement.

## 3.3 TrendEngine

Le TrendEngine calcule un signal de tendance scalaire sur le benchmark (SPY)
en combinant deux composantes :

**Signal Donchian.** Pour une fenêtre *don_win* :

```
d_t = +1  si P_t ≥ max(P_{t-don_win}, ..., P_{t-1})
d_t = -1  si P_t ≤ min(P_{t-don_win}, ..., P_{t-1})
d_t =  0  sinon
```

**Signal de trend-following.** Pour une fenêtre *tf_win* :

```
tf_t = sign(P_t - MA_{tf_win}(P))  ×  |P_t - MA_{tf_win}(P)| / σ_{20}(P)
```

où σ_{20}(P) est la volatilité réalisée sur 20 jours et MA_{tf_win} la
moyenne mobile exponentielle de fenêtre tf_win.

**Signal combiné :**

```
trend_t = clip(w_don × d_t + w_tf × tf_t, -1, 1)
```

Le TrendEngine est entièrement rule-based — aucun paramètre n'est appris.
Les fenêtres don_win et tf_win, ainsi que les poids de fusion, sont optimisés
par Optuna sur la période 2020–2021.

Les valeurs optimales obtenues dans SCAF v4.1 sont don_win=5, tf_win=50,
ce qui correspond à un signal réactif à courte mémoire — cohérent avec
l'environnement de trading 2022–2024 caractérisé par des tendances fortes
et des retournements rapides.

## 3.4 HybridSignal et fusion

Le module HybridSignal combine le signal directionnel (TrendEngine) et
le signal de ranking sectoriel (TCCA/CSR) en un P&L journalier :

```
pnl_raw_t = w_trend × pnl_trend_t + w_cs × pnl_cs_t
```

**Composante trend :** le signal trend_t est directement utilisé comme
position sur le benchmark SPY, avec les rendements log journaliers de SPY
comme base de calcul.

**Composante cross-sectorielle :** les poids cs_weights_t (issus du TCCA
via CrossSectionPortfolio) définissent un portefeuille long/short sur les
18 secteurs, avec positions longues sur les actifs à score élevé et positions
courtes sur les actifs à faible score.

Les poids de fusion w_trend et w_cs sont optimisés par Optuna. Les valeurs
obtenues (w_trend ≈ 0,97–0,99, w_cs ≈ 0,01–0,05) confirment que le TrendEngine
constitue la source d'alpha dominante, avec la composante sectorielle apportant
une diversification marginale.

## 3.5 RiskManager

Le RiskManager applique deux mécanismes de contrôle du risque en cascade :

**Ciblage de volatilité (VolTargeting).** Le signal brut est scalé par le
rapport entre la volatilité cible (target_vol) et la volatilité réalisée
sur vol_window jours :

```
scale_t = target_vol / max(σ_{vol_window}(ret), ε)
pnl_scaled_t = pnl_raw_t × clip(scale_t, 0, 3)
```

**Tiers de drawdown.** Trois paliers progressifs réduisent l'exposition
lorsque le drawdown dépasse des seuils dd1 < dd2 < dd3 :

```
                    dd > dd1 → exposition × dd_s1
                    dd > dd2 → exposition × dd_s2
                    dd > dd3 → exposition × dd_s3
```

Les valeurs optimales (dd1=0,04, dd2=0,09, dd3=0,17 dans v4.1) définissent
une réduction progressive et automatique de l'exposition lors des périodes
de pertes, ce qui explique le MaxDD quasi-nul (−0,04%) malgré des rendements
élevés.

## 3.6 Protocole d'optimisation multi-agents

L'espace de paramètres complet de SCAF v4.1 comprend 15 hyperparamètres
continus et discrets. Pour l'explorer efficacement avec des garanties de
couverture, nous proposons un protocole d'optimisation multi-agents organisé
en quatre catégories spécialisées :

| Catégorie | Paramètres optimisés | N agents |
|---|---|---|
| `strategy_params` | don_win, tf_win, w_trend, w_cs | 1 000 |
| `ml_params` | ml_thr, horizon | 1 000 |
| `risk_params` | target_vol, vol_window, dd1–dd3 | 500 |
| `portfolio_blend` | w_trend, w_cs, max_pos | 500 |

Chaque catégorie opère sur un Optuna Study partagé via l'algorithme TPE
(Tree-structured Parzen Estimator). Les essais des autres catégories
alimentent la distribution probabiliste de chaque catégorie, permettant une
exploration coopérative de l'espace complet. L'élagage médian (MedianPruner)
interrompt les essais non-prometteurs après 15 évaluations, réduisant le
coût computationnel total.

Cette organisation en catégories spécialisées produit une meilleure couverture
de chaque sous-espace fonctionnel qu'une recherche uniforme : chaque catégorie
peut consacrer plus d'essais aux interactions entre paramètres fonctionnellement
liés (ex. : don_win et tf_win sont co-optimisés dans `strategy_params`).

## 3.7 Suppression du RegimeFilter (v4.1 / v5)

L'architecture originale (v4) incluait un RegimeFilter basé sur un ensemble
de classificateurs (LogisticRegression, BaggingClassifier, HistGradientBoosting)
prédit l'état de marché bull/bear/sideways pour moduler le signal de tendance.
L'étude d'ablation révèle que ce composant dégrade la performance (−44% Sharpe)
en raison de son AUC quasi-aléatoire (0,52). SCAF v4.1 supprime ce composant
et fixe les scalaires de régime s_bull = s_bear = s_side = 1,0.

L'intuition est la suivante : sur un univers d'ETFs sectoriels avec signal de
tendance réactif (don_win=5), les transitions de régime sont implicitement
gérées par le TrendEngine via ses fenêtres courtes, rendant tout filtre de
régime explicite redondant et nuisible. Le TCCA (v5) réalise un conditionnement
implicite plus puissant via l'attention, sans nécessiter de classification
discrète du régime.
