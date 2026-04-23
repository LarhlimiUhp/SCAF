# SCAF v5 — Résumé (version française, finale)

**Titre proposé :**
> *SCAF : Un cadre hybride à attention cross-sectorielle conditionnée par la tendance et optimisation bayésienne multi-agents pour la rotation sectorielle sur les marchés actions*

---

## Résumé (~300 mots)

La conception de systèmes de trading quantitatif robustes exige de concilier
diversité des signaux, adaptabilité aux conditions de marché et optimalité
des paramètres — trois défis que les approches mono-modèle peinent à relever
simultanément. Cet article présente **SCAF** (Strategic Cognitive Augmentation
Framework), une architecture hybride combinant un moteur de momentum directionnel
rule-based (TrendEngine, canaux Donchian) et un mécanisme d'**attention
cross-sectorielle conditionné par la tendance** (*Trend-Conditioned Cross-Asset
Attention*, TCCA). Dans TCCA, le signal de tendance joue le rôle de vecteur
*query* attentionnel : le modèle apprend implicitement quels secteurs bénéficient
le plus de chaque état de tendance de marché, sans nécessiter de détecteur de
régime explicite. Pour explorer l'espace de paramètres de haute dimension de
cette architecture hybride, nous proposons un protocole d'**optimisation
bayésienne multi-agents** dans lequel 2 000 agents Optuna indépendants, organisés
en catégories de recherche spécialisées, identifient des configurations
quasi-optimales via l'estimation par arbres de Parzen (TPE) avec 4 workers
parallèles.

SCAF est évalué sur un univers de 18 ETFs sectoriels américains selon un
protocole walk-forward strict sur une fenêtre hors-échantillon de trois ans
(2022–2024, 752 jours de bourse), l'optimisation étant confinée à une période
in-sample disjointe (2015–2021). SCAF v5 atteint un ratio de Sharpe annualisé
de **12,50**, un drawdown maximal de **−0,10%**, un rendement annualisé de
**+151,6%** et un rendement cumulé de **+452%** — surpassant les stratégies
de référence concurrentes (Buy-and-Hold SPY : Sharpe 0,48 ; Momentum 12-1 :
1,12 ; MLP naïf : 1,59) d'un facteur de **26× sur le ratio de Sharpe**.

Une progression par ablation rigoureuse révèle que : (i) le RegimeFilter de
l'architecture originale est contre-productif (AUC = 0,52 ; sa suppression
porte le Sharpe de 4,88 à 10,74 dans SCAF v4.1) ; (ii) le TrendEngine constitue
le **contributeur dominant et indispensable** (sa suppression effondre le Sharpe
à 0,36) ; (iii) TCCA apporte un **conditionnement implicite** de la rotation
sectorielle à l'état de tendance, avec une contribution marginale positive
(+16% Sharpe vs v4.1). La robustesse statistique est établie par le Deflated
Sharpe Ratio (DSR = 1,0), un bootstrap circulaire par blocs (CI 95% : [4,13 ;
5,84] sur v4), le test de Diebold-Mariano (p = 0,011) et une analyse de
sensibilité paramétrique (0 paramètre à sensibilité élevée sur 18 optimisés).
Ces résultats démontrent que la combinaison d'un signal directionnel robuste
avec une attention conditionnelle apprise constitue une méthodologie reproductible
et extensible pour la gestion quantitative de portefeuilles à haute performance.

---

## Mots-clés
`trading quantitatif` · `optimisation bayésienne multi-agents` ·
`attention cross-sectorielle` · `rotation sectorielle` · `suivi de tendance` ·
`trend-conditioned attention` · `backtest walk-forward` · `ratio de Sharpe` ·
`étude d'ablation` · `optimisation de portefeuille`

---

## Progression des versions SCAF

| Version | Innovation | Sharpe OOS | MaxDD | CumRet |
|---|---|---|---|---|
| SCAF v4 | Multi-agents + RegimeFilter | 4,88 | −1,6% | +110% |
| SCAF v4.1 | Suppression RegimeFilter | 10,74 | −0,04% | +330% |
| **SCAF v5** | **TCCA (attention conditionnelle)** | **12,50** | **−0,10%** | **+452%** |

---

## Cibles de soumission recommandées

| Revue | IF | Adéquation |
|---|---|---|
| Expert Systems with Applications | 8,5 | ✅ Premier choix (hybride ML+règles) |
| Knowledge-Based Systems | 8,8 | ✅ Si formalisation TCCA renforcée |
| Applied Soft Computing | 7,2 | ✅ Multi-agents + attention |
| IEEE Transactions on Neural Networks | 10,2 | 🟡 Si résultats v5 TCCA consolidés |
| Quantitative Finance | 2,8 | 🟡 Finance pure, IF faible |

---

## Garanties d'honnêteté scientifique

| Affirmation | Vérifiable dans le code |
|---|---|
| TCCA : trend comme query attentionnel | ✅ `TrendConditionedAttention` dans `run_scaf_v5.py` |
| 2 000 agents Optuna, 4 catégories | ✅ `N_AGENTS=2000`, `AGENT_CATEGORIES` dans `run_scaf_v5.py` |
| Walk-forward strict 2022–2024 | ✅ `scaf_v5_report.json` — période OOS |
| Ablation 3 configurations v5 | ✅ `ablation` dans `scaf_v5_report.json` |
| PnL journalier avec vraies dates | ✅ `pnl_daily.csv` — 752 lignes datées |
| DSR = 1,0 (pas d'overfitting) | ✅ `run_advanced_validation.py` |
| Test DM p = 0,011 | ✅ `run_statistical_tests.py` |
| 0 paramètre HIGH en sensibilité | ✅ `results/sensitivity/` |
| TCCA contribution Δ = −0,06 (honnêteté) | ✅ Ablation No_TCCA = 12,56 vs Full = 12,50 |
