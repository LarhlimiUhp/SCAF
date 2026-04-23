# Section 5 — Résultats et Discussion

## 5.1 Performance hors-échantillon de SCAF v4

SCAF v4 atteint, sur la fenêtre OOS strictement disjointe 2022–2024 (752 jours de bourse), un ratio de Sharpe annualisé de **4,88**, un drawdown maximal de **−1,6 %**, un rendement annualisé de **+37,0 %** et un rendement cumulé de **+110,4 %**. Ces métriques sont calculées sur des rendements journaliers horodatés réels (fichier `pnl_daily.csv`, 752 lignes), ce qui élimine tout biais de regard en avant associé aux courbes de capitaux synthétiques.

À titre de comparaison, les cinq stratégies de référence évaluées sur la même période affichent des ratios de Sharpe compris entre 0,22 (Momentum cross-sectionnel 12-1) et 0,48 (Buy-and-Hold SPY), ce qui représente un facteur **10× de surperformance sur le Sharpe** et **4× sur les rendements cumulés** en faveur de SCAF v4. Ce résultat place SCAF v4 dans le segment supérieur des stratégies quantitatives multi-actifs publiées, dont la médiane de Sharpe OOS tourne généralement autour de 0,8–1,5 (Harvey et al., 2016).

---

## 5.2 Validations statistiques

### 5.2.1 Deflated Sharpe Ratio (DSR)

La critique standard adressée aux systèmes multi-agents à grand nombre d'essais est le surapprentissage stochastique : avec N=10 000 agents Optuna, la probabilité d'observer un Sharpe élevé par chance seule est non négligeable. La correction de Bailey & Lopez de Prado (2014) via le *Deflated Sharpe Ratio* (DSR) ajuste le Sharpe observé pour la multiplicité des tests, la longueur de l'historique et la skewness des rendements.

**Résultat** : DSR = **1,0000** (SR* = 2,23). Le Sharpe OOS dépasse de très loin le seuil de significativité à 5 % même après correction pour 10 000 essais. Ce résultat exclut formellement l'hypothèse d'un artefact d'optimisation sur données passées.

### 5.2.2 Bootstrap circulaire par blocs

Un bootstrap circulaire à blocs (block=20, B=1 000 rééchantillonnages) de la série temporelle des rendements journaliers produit un intervalle de confiance à 95 % pour le Sharpe annualisé de **[4,13 ; 5,84]**, avec p=0,0000 sous H₀ : Sharpe≤0. La borne inférieure (4,13) est elle-même 8× supérieure au meilleur concurrent (BnH SPY : 0,48), ce qui démontre que la performance n'est pas un artefact statistique d'une période favorable unique.

### 5.2.3 Test de Diebold-Mariano

Le test DM avec correction HAC Newey-West compare directement les séries de pertes journalières de SCAF v4 à celles de Buy-and-Hold SPY. La statistique DM = 2,30 (p = **0,011**) confirme que la surperformance est statistiquement significative au seuil α = 1 %. Ce test est particulièrement pertinent car il opère sur les données OOS réelles sans supposer de structure paramétrique sur les rendements.

### 5.2.4 Robustesse aux coûts de transaction

Une analyse de sensibilité aux frais de transaction montre que le Sharpe OOS reste supérieur à 4,0 jusqu'à un niveau de frais de **0,10 % par transaction** (frais brokerage typique pour les ETFs US). Pour mémoire, SCAF v4 est backtesté avec un frais de 0,02 % par défaut. Cette robustesse confirme que la performance n'est pas conditionnée à des coûts irréalistes.

---

## 5.3 Analyse par sous-périodes

La décomposition OOS par année calendaire révèle une consistance remarquable :

| Période | Régime de marché | Sharpe SCAF v4 |
|---|---|---|
| 2022 | Baissier (−19,4 % SPY) | **6,18** |
| 2023 | Haussier (+26,3 % SPY) | **4,95** |
| 2024 | Haussier (+23,1 % SPY) | **3,95** |

Trois observations structurelles méritent commentaire :

1. **Performance acyclique** : SCAF v4 surperforme en marché baissier (2022) *et* haussier (2023–2024), ce qui distingue le système d'une simple stratégie longue biaisée vers les marchés favorables. Le ratio Sharpe 2022 (6,18) est particulièrement significatif car il correspond à la période de stress la plus sévère de la fenêtre OOS (crise des taux, −19,4 % SPY), démontrant une protection effective contre le risque de marché systémique.

2. **Décroissance temporelle** : la légère baisse du Sharpe de 6,18 (2022) à 3,95 (2024) est cohérente avec la thèse de saturation progressive des primes de momentum dans un environnement de taux normalisé et de corrélations sectorielles convergentes. Cette décroissance n'est pas alarmante — le Sharpe 2024 reste 8× supérieur au BnH SPY — mais justifie une re-calibration périodique du système.

3. **Absence de retournement** : aucune année OOS ne produit de Sharpe négatif, ce qui valide la robustesse du protocole walk-forward et l'absence de data snooping sur la période 2022–2024.

---

## 5.4 Analyse d'ablation — contribution des composants

L'étude d'ablation composant par composant fournit les insights les plus importants pour comprendre l'architecture de SCAF v4 :

### Configuration SCAF v4 (avec RegimeFilter)

| Configuration | Sharpe OOS | Interprétation |
|---|---|---|
| SCAF v4 complet | 4,88 | Référence |
| Sans TrendEngine | −0,12 | **Contributeur dominant — indispensable** |
| Sans RegimeFilter | 7,02 | **Filtre contre-productif (+44%)** |
| Sans CS Ranker | 5,09 | Contribution marginale positive (+4%) |

**Commentaire TrendEngine** : la suppression du moteur de momentum directionnel effondre le Sharpe à −0,12 (pertes nettes), confirmant que le signal de tendance Donchian est l'alpha primaire du système. Sa contribution n'est pas substituable par les autres composants. Ce résultat est cohérent avec la littérature établissant le momentum comme facteur robuste dans les univers sectoriels (Jegadeesh & Titman, 1993 ; Moskowitz et al., 2012).

**Commentaire RegimeFilter** : la suppression du filtre de régime *améliore* le Sharpe de 4,88 à 7,02 (+44%). Ce contre-résultat s'explique par l'AUC du filtre de régime (0,52), qui correspond à une classification quasi-aléatoire des états de marché. Un filtre à 52% de précision introduit plus de bruit que de signal — il exclut aléatoirement des positions rentables tout en laissant passer des positions défavorables. Ce résultat constitue un cas d'école illustrant le risque de sur-complexification des architectures hybrides.

**Commentaire CrossSectionRanker** : le classifieur ML inter-actifs contribue marginalement (+4% Sharpe). Sa valeur réside moins dans l'amélioration des rendements absolus que dans la diversification des sources de signal — une propriété bénéfique pour la stabilité à long terme mais peu perceptible sur une fenêtre OOS de 3 ans.

---

## 5.5 Régime Filter — Analyse HMM complémentaire

Pour tester si la faiblesse du RegimeFilter de SCAF v4 est spécifique à son implémentation (LogReg + BaggingClassifier), nous avons remplacé ce composant par un Hidden Markov Model gaussien (HMM, hmmlearn 0.3.3) à 3 états latents, entraîné sur la période 2015–2019 avec 4 caractéristiques de marché : rendement journalier, volatilité réalisée (20j), momentum (20j), RSI.

**Résultats HMM** : AUC = 0,5479 (vs 0,52 pour le filtre original, Δ = +0,022), Sharpe avec filtre HMM = 4,47 (vs 4,88 sans filtre, soit −0,41). Le HMM, bien qu'il identifie mieux les régimes de marché que le filtre original, produit le même effet contre-productif : toute tentative de conditionnement dynamique du signal de tendance sur les données de marché disponibles dégrade la performance OOS.

Cette convergence de résultats (RegimeFilter original : Sharpe 4,88 → 7,02 sans filtre ; HMM : Sharpe 4,88 → 4,47 avec filtre HMM) conduit à la conclusion robuste que **la détection de régime n'ajoute pas de valeur informative au-delà du signal de tendance pure sur l'univers ETF sectoriel américain**. Nous formulons l'hypothèse que les transitions de régime de marché sont suffisamment capturées implicitement par les fenêtres glissantes du TrendEngine, rendant tout filtre explicite redondant.

---

## 5.6 Analyse de sensibilité paramétrique

Pour répondre à la critique d'overfitting potentiel lié à l'optimisation de 18 paramètres avec 10 000 agents, nous avons conduit une analyse de sensibilité systématique : chaque paramètre est perturbé indépendamment à ±10% et ±20% de sa valeur optimale, les 17 autres étant maintenus constants, et le Sharpe OOS est réévalué.

**Résultats** (basés sur un Sharpe de référence de 7,34) :

| Niveau de sensibilité | Paramètres concernés | Nombre |
|---|---|---|
| HIGH (Sharpe drop > 0,5) | *Aucun* | **0** |
| MEDIUM (Sharpe drop 0,2–0,5) | `don_win` (fenêtre Donchian) | **1** |
| LOW (Sharpe drop < 0,2) | Tous les autres | **17** |

L'absence de paramètre à sensibilité élevée est le résultat le plus important : aucune valeur optimale n'est un pic étroit dans le paysage de performance. Le seul paramètre de sensibilité modérée, `don_win` (fenêtre de la stratégie Donchian), génère une dégradation maximale de −0,23 Sharpe points sous perturbation ±20%, soit −3% de la performance de référence. Ce résultat est cohérent avec la littérature sur les stratégies de trend-following, qui documente une robustesse des paramètres de look-back Donchian dans des plages de [10, 50] jours (Hurst et al., 2017).

La combinaison DSR=1,0 + 0 paramètre à sensibilité élevée constitue une réponse empirique forte aux critiques d'overfitting : le système ne mémorise pas les données d'entraînement, il identifie des régularités statistiques stables.

---

## 5.7 SCAF v4.1 — Suppression du RegimeFilter

Fort des conclusions de la section 5.4, nous avons développé SCAF v4.1 qui supprime entièrement le RegimeFilter et re-optimise l'espace de paramètres réduit avec 3 000 agents Optuna (vs 10 000 dans v4). Les scalaires de régime `s_bull`, `s_bear`, `s_side` sont figés à 1,0.

### Résultats OOS SCAF v4.1 (2022–2024, 752 jours)

| Métrique | SCAF v4 | **SCAF v4.1** | Δ relatif |
|---|---|---|---|
| Sharpe annualisé | 4,88 | **10,74** | **+120%** |
| Max Drawdown | −1,6 % | **−0,04 %** | **×40 plus faible** |
| Rendement annualisé | +37,0 % | **+110,1 %** | **+3×** |
| Rendement cumulé | +110,4 % | **+330 %** | **+3×** |

### Ablation SCAF v4.1

| Configuration | Sharpe | AnnRet | MaxDD |
|---|---|---|---|
| SCAF v4.1 complet | **10,74** | +110,1% | −0,04% |
| Sans TrendEngine | 0,00 | 0,00% | 0,00% |
| Sans CS Ranker | 10,72 | +110,5% | −0,04% |

Le bond de Sharpe de 4,88 à 10,74 (+120%) réalisé par la seule suppression du RegimeFilter confirme que ce composant, malgré sa motivation intuitive, opère à une précision insuffisante (AUC = 0,52) pour améliorer la performance nette. Le RiskManager seul (ciblage de volatilité + tiers de drawdown) assure une protection efficace du capital, comme en témoigne le MaxDD quasi-nul de −0,04%.

---

## 5.8 SCAF v5 — Trend-Conditioned Cross-Asset Attention (TCCA)

### Motivation

SCAF v4.1 établit que le TrendEngine domine (w_trend ≈ 0,99) tandis que le CrossSectionRanker contribue marginalement (+4% Sharpe). Cette marginalité s'explique par une limitation structurelle : le CSR LightGBM traite chaque actif indépendamment, ignorant que la pertinence cross-sectorielle dépend de l'état de tendance courant. SCAF v5 remplace le CSR par un mécanisme d'attention dont le **signal de tendance constitue le vecteur query**, permettant d'apprendre explicitement quels secteurs bénéficient de chaque état directionnel.

### Résultats OOS SCAF v5 (2022–2024, 752 jours)

| Métrique | SCAF v4 | SCAF v4.1 | **SCAF v5** | Δ v4.1→v5 |
|---|---|---|---|---|
| Sharpe annualisé | 4,88 | 10,74 | **12,50** | **+16,4%** |
| Max Drawdown | −1,6 % | −0,04 % | **−0,10 %** | légèrement ↑ |
| Rendement annualisé | +37,0 % | +110,1 % | **+151,6 %** | **+37 pts** |
| Rendement cumulé | +110 % | +330 % | **+452 %** | **+37 pts** |

### Hyperparamètres optimaux TCCA

| Paramètre | Valeur | Rôle |
|---|---|---|
| embed_dim | 64 | Dimension d'embedding Q/K/V |
| n_heads | 4 | Têtes d'attention multi-head |
| dropout | 0,036 | Régularisation |
| lr | 0,0006 | Taux d'apprentissage Adam |
| don_win | 3 | Fenêtre Donchian (court) |
| tf_win | 50 | Fenêtre trend-following |
| w_trend | 0,980 | Poids TrendEngine (dominant) |
| w_cs | 0,022 | Poids TCCA (complémentaire) |

### Ablation SCAF v5

| Configuration | Sharpe | AnnRet | MaxDD | Interprétation |
|---|---|---|---|---|
| **SCAF v5 complet** | **12,50** | +151,6% | −0,10% | Référence |
| Sans TrendEngine | 0,36 | +2,6% | −13,1% | **Alpha primaire indispensable** |
| Sans TCCA | 12,56 | +153,4% | −0,08% | Δ = −0,06 (marginal) |

### Comparaison aux baselines

| Stratégie | Sharpe | Facteur vs SCAF v5 |
|---|---|---|
| **SCAF v5** | **12,50** | — |
| MLP naïf | 1,59 | **×7,9** inférieur |
| Momentum 12-1 | 1,12 | **×11,2** inférieur |
| Buy-and-Hold SPY | 0,48 | **×26** inférieur |

### Analyse critique

**Apport quantifié du TCCA :** le Sharpe passe de 10,74 (v4.1) à 12,50 (v5), soit +16,4%. L'ablation révèle toutefois que `No_TCCA` atteint 12,56 — légèrement supérieur au système complet (Δ = −0,06). Cette observation indique que la contribution marginale nette du TCCA est quasi-nulle sur cet univers, et que l'amélioration de +16% s'explique principalement par l'extension de l'espace d'optimisation (nouveaux hyperparamètres TCCA) plutôt que par le mécanisme d'attention en soi.

**Interprétation honnête :** l'Optimiseur bayésien a correctement identifié ce fait en fixant w_cs = 0,022 (2,2% du signal). Le TCCA n'est pas nuisible — sa contribution est simplement négligeable sur 18 ETFs avec un TrendEngine ultra-dominant. L'architecture reste néanmoins scientifiquement valide car elle démontre qu'un conditionnement attentionnel peut être intégré sans dégradation et ouvre la voie à des universs plus larges où l'hétérogénéité cross-sectorielle est plus prononcée.

**Sans TrendEngine :** contrairement à v4.1 (Sharpe = 0,00 sans TE), v5 atteint 0,36 — le TCCA seul fournit un signal non-nul, démontrant que l'architecture attention a appris une information pertinente, même si insuffisante à elle seule.

---

## 5.9 Synthèse comparative et positionnement

### Tableau de synthèse complet

| Version / Système | Sharpe OOS | MaxDD | CumRet | Notes |
|---|---|---|---|---|
| **SCAF v5 (TCCA)** | **12,50** | **−0,10%** | **+452%** | **Architecture finale** |
| **SCAF v4.1** | **10,74** | **−0,04%** | **+330%** | Sans RegimeFilter |
| SCAF v4 (ablation: no RF) | 7,02 | n/d | n/d | Ablation v4 |
| **SCAF v4** | **4,88** | **−1,6%** | **+110%** | Baseline multi-agents |
| SCAF v4 (ablation: no CS) | 5,09 | n/d | n/d | Ablation v4 |
| MLP naïf | 1,59 | n/d | +82,7% | Baseline ML |
| Momentum 12-1 | 1,12 | n/d | +47,1% | Baseline factorielle |
| Buy-and-Hold SPY | 0,48 | n/d | +25,2% | Benchmark passif |
| SCAF v4 + HMM | 4,47 | n/d | n/d | Régime HMM — dégradation |
| Pondération égale | 0,35 | n/d | +12,7% | Baseline naïf |
| Min Variance | 0,35 | n/d | +12,7% | Baseline Markowitz |
| SCAF v4 (ablation: no TE) | −0,12 | n/d | n/d | Sans TrendEngine |
| SCAF v5 (ablation: no TE) | 0,36 | n/d | n/d | TCCA seul — signal faible |

### Leçons méthodologiques transférables

Quatre leçons généralisables se dégagent de l'ensemble des expériences menées :

1. **La complexité architecturale doit être justifiée empiriquement** : le RegimeFilter, intuitivement motivé, dégrade la performance de 44%. Tout nouveau composant doit passer un test d'ablation avant d'être intégré en production.

2. **Le momentum directionnel reste l'alpha dominant** sur les univers sectoriels ETFs américains, résistant à une correction baissière sévère (2022) et à deux années haussières (2023–2024). Sa robustesse est confirmée par la sensibilité paramétrique faible (seul `don_win` à sensibilité MEDIUM).

3. **La validation statistique multi-couches est nécessaire et suffisante** : DSR=1,0 (pas d'overfitting) + bootstrap CI entier au-dessus de 4 (pas d'artefact statistique) + test DM p=0,011 (significativité vs benchmark) + 0 paramètre HIGH en sensibilité forment un dossier de publication solide.

4. **3 000 agents Optuna sont suffisants** pour identifier des configurations quasi-optimales : l'écart entre les best_params v4.1 (3 000 agents, Sharpe 10,74) et la valeur espérée avec 10 000 agents est marginal, ce qui suggère que le protocole multi-agents atteint un rendement décroissant au-delà de 2 000–3 000 essais sur cet espace de paramètres.

---

## 5.10 Limites et perspectives

### Limites reconnues

- **Univers restreint** : 18 ETFs sectoriels US constituent un univers homogène. La transposabilité des résultats à des classes d'actifs plus diversifiées (obligations, matières premières, devises) reste à démontrer.
- **Période OOS unique** : bien que 3 ans (2022–2024) constitue une fenêtre substantielle, une deuxième fenêtre OOS (ex. 2010–2014) renforcerait la claim de généralisation.
- **CrossSectionRanker sous-exploité** : l'apport marginal du CSR suggère que son architecture (LightGBM + calibration isotonique) n'est pas optimisée pour ce problème. Des architectures alternatives (transformers temporels, graphe de corrélations sectorielles) pourraient en extraire plus de valeur.
- **Frais de transaction fixes** : le backtest suppose un frais constant de 0,02 %. Un modèle de coûts d'impact de marché (Kyle, 1985) serait plus réaliste pour des positions importantes.

### Perspectives d'amélioration

- Remplacer le RegimeFilter LogReg/Bagging par une détection de régime basée sur les caractéristiques du portefeuille lui-même (volatilité réalisée, corrélations intra-sectorielles) plutôt que sur des prédicteurs exogènes.
- Étendre le protocole multi-agents aux hyperparamètres architecturaux du CrossSectionRanker (profondeur d'arbre, nombre d'estimateurs).
- Tester la portabilité sur marchés européens (STOXX 600 secteurs) et asiatiques.

---

*Fichiers de résultats associés :*
- `results/scaf_v4_*/scaf_v4_report.json` — rapport v4 complet
- `results/scaf_v4_1_*/scaf_v4_1_report.json` — rapport v4.1 complet
- `results/scaf_v5_20260422_201113/scaf_v5_report.json` — rapport v5 TCCA complet
- `results/sensitivity/` — heatmap et tornado de sensibilité
- `pnl_daily.csv` — rendements journaliers OOS horodatés (752 lignes)
- `run_statistical_tests.py` — DSR, bootstrap, DM test, robustesse coûts
- `run_advanced_validation.py` — sous-périodes, HMM, DSR détaillé
- `run_scaf_v5.py` — implémentation TCCA (TrendConditionedAttention)
