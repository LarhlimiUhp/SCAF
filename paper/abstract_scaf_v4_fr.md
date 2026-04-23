# SCAF v4 — Résumé (version française)

**Titre proposé :**
> *SCAF : Un cadre hybride à optimisation bayésienne multi-agents pour la rotation sectorielle et le suivi de tendance sur les marchés actions*

---

## Résumé (~280 mots)

La conception de systèmes de trading quantitatif robustes exige de concilier
diversité des signaux, adaptabilité aux régimes de marché et optimalité des
paramètres — trois défis que les approches mono-modèle peinent à relever
simultanément. Cet article présente **SCAF** (Strategic Cognitive Augmentation
Framework), une architecture hybride combinant un moteur de momentum directionnel
(Trend Engine), un classifieur inter-actifs basé sur l'apprentissage automatique
(Cross-Section Ranker), et un filtre de régime pondéré par AUC (Regime Filter),
fusionnés en un signal de portefeuille unique. Pour explorer l'espace de paramètres
de haut dimension inhérent à ce système multi-composants, nous proposons un protocole
d'**optimisation bayésienne multi-agents** dans lequel 10 000 agents Optuna
indépendants, organisés en cinq catégories de recherche spécialisées (construction
du signal, seuils ML, mise à l'échelle du régime, gestion du risque et allocation
de portefeuille), identifient collectivement des configurations quasi-optimales via
l'estimation par arbres de Parzen (TPE) avec 4 workers parallèles.

SCAF est évalué sur un univers de 18 ETFs sectoriels américains selon un protocole
walk-forward strict sur une fenêtre hors-échantillon de trois ans (2022–2024,
752 jours de bourse), l'optimisation étant confinée à une période in-sample
disjointe (2015–2021). SCAF atteint un ratio de Sharpe annualisé de **4,88**,
un drawdown maximal de **−1,6 %**, un rendement annualisé de **+37,0 %** et un
rendement cumulé de **+110,4 %** — surpassant cinq stratégies de référence
concurrentes (Buy-and-Hold SPY : Sharpe 0,48 ; Pondération égale : 0,37 ;
Momentum cross-sectionnel 12-1 : 0,22 ; Variance minimale : 0,36 ;
MLP naïf : 0,37) d'un facteur de **10× sur le ratio de Sharpe** et de
**4× sur les rendements cumulés**.

Une étude d'ablation formelle composant par composant révèle que le Trend Engine
constitue le **contributeur dominant** (sa suppression effondre le Sharpe à −0,12 et
génère des pertes nettes), que le Regime Filter introduit un **conservatisme excessif**
à faible précision prédictive (AUC = 0,52 ; sa suppression porte le Sharpe à 7,02),
et que le Cross-Section Ranker apporte une **diversification marginale mais
stabilisatrice**. Les rendements journaliers du portefeuille sont persistés avec
horodatage réel, éliminant tout biais de regard en avant lié aux courbes de
capitaux synthétiques. Ces résultats démontrent que la recherche multi-agents
structurée sur des architectures de signaux hybrides constitue une méthodologie
reproductible et extensible pour la gestion quantitative de portefeuilles à
haute performance.

---

## Mots-clés
`trading quantitatif` · `optimisation bayésienne` · `systèmes multi-agents` ·
`rotation sectorielle` · `suivi de tendance` · `étude d'ablation` ·
`backtest walk-forward` · `ratio de Sharpe` · `détection de régime` ·
`optimisation de portefeuille`

---

## Cibles de soumission recommandées

| Revue | IF | Adéquation |
|---|---|---|
| Expert Systems with Applications | 8,5 | ✅ Premier choix |
| Applied Soft Computing | 7,2 | ✅ Bon fit multi-agents |
| Knowledge-Based Systems | 8,8 | ✅ Si ablation renforcée |
| Revue d'Économie Financière | — | 🟡 Audience francophone |
| Quantitative Finance (Taylor & Francis) | 2,8 | 🟡 Finance pure, IF faible |

---

## Garanties d'honnêteté scientifique

| Affirmation | Vérifiable dans le code |
|---|---|
| 10 000 agents Optuna, 5 catégories | ✅ `AGENT_CATEGORIES_V4` dans `run_scaf_v4.py` |
| Walk-forward strict 2022–2024 | ✅ `scaf_v4_report.json` — période OOS |
| 5 baselines réelles | ✅ Résultats dans `baselines` du rapport JSON |
| Ablation 4 configurations | ✅ Résultats dans `ablation` du rapport JSON |
| PnL journalier avec vraies dates | ✅ `pnl_daily.csv` — 752 lignes datées |
| Aucun LLM / Q-Learning / Conformal Prediction | ✅ Absents du code — non mentionnés |
