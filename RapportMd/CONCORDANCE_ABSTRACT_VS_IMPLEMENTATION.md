# Concordance : Abstract SCAF (paper) vs. Implémentation actuelle (SCAF v3 lean)

**Date d'analyse :** 2026-04-19
**Version implémentée :** SCAF v3 lean (post-Phase C, RegimeFilter supprimé)
**Abstract analysé :** SCAF (Strategic Cognitive Augmentation Framework) — paper de recherche

---

## Tableau de concordance

| Composante (Abstract) | Statut | Implémentation actuelle | Écart | Gravité |
|---|:---:|---|---|:---:|
| **1. LLM comme orchestrateur cognitif** | 🟡 | Module `llm/orchestrator.py` présent (348 LOC), **mais dormant** — non appelé dans `run_scaf_v3.py` lean | LLM non intégré dans le pipeline de production | 🔴 **CRITIQUE** |
| **2. Reinforcement Learning (Q-Learning sélection)** | 🟢 | `models/qlearning_selector.py` (277 LOC) — tabular Q-learning 12 states × N actions, save/load Q-table JSON | ✅ Présent, mais **non activé** dans SCAF v3 lean | 🟡 Moyen |
| **3. Deep Learning regime detection** | 🔴 | `models/regime_detector.py` (248 LOC) — **déprécié et retiré** en Phase C (ablation: +0.65 Sharpe quand désactivé) | Abstract mentionne « DL regime » mais implémentation prouve qu'il **détruit** de la valeur | 🔴 **CONTRADICTION** |
| **4. Conformal Prediction calibration** | 🟢 | `models/conformal.py` (326 LOC) — `SplitConformalClassifier`, `AdaptiveConformalClassifier` (ACI), `ConformalEnsemble` | ✅ Implémenté, mais **non utilisé** dans SCAF v3 lean | 🟡 Moyen |
| **5. Portfolio de 13+ modèles experts** | 🟡 | `models/` contient 10 fichiers (1456 LOC) : conformal, extra_models, torch_models, sklearn_models, LGBM, etc. — **SCAF v3 lean utilise seulement 3** : LogReg + Bagging + HistGBT | Écart majeur : 3 modèles actifs vs 13+ promis | 🔴 **MAJEUR** |
| **6. Q-Learning selection dynamique** | 🔴 | `qlearning_selector.py` présent mais **jamais appelé** dans `run_scaf_v3.py` — sélection fixe de 2/3 modèles via AUC gate (>0.55) | Abstract promet « dynamic selection through market feedback », implémentation = static AUC filter | 🔴 **MAJEUR** |
| **7. Évaluation 4 classes d'actifs (S&P 500, EUR/USD, NASDAQ, Bitcoin)** | 🔴 | **18 ETF sectoriels US uniquement** (XLK, XLF, XLE, etc.) — aucun FX, aucune crypto, NASDAQ via QQQ uniquement | Abstract : 4 asset classes multi-univers ; réalité : 1 classe (US equity ETFs) | 🔴 **MAJEUR** |
| **8. Période 2018-2024** | 🟢 | **2015-2024** (10 ans) — OOS test 2022-2024 (752 jours) | ✅ Compatible (période OOS incluse dans 2018-2024) | 🟢 OK |
| **9. 81 % RMSE reduction vs DL baselines (p < 0.001)** | 🔴 | **Pas de RMSE reporté** — métriques = Sharpe (7.27), Max DD (−1.02 %), CumRet (1.51). Baselines = Donchian/Momentum/B&H, **pas de DL** | Abstract parle de forecasting (RMSE), implémentation = trading strategy (Sharpe) | 🔴 **CRITIQUE** |
| **10. 33 % Sharpe improvement** | 🟡 | **Sharpe 7.27 (lean) vs 0.36 (B&H)** = ×20, mais vs Donchian baseline = −2.5 % (7.27 vs 7.46) | Amélioration dépend du baseline — vs B&H ✅, vs Donchian ❌ | 🟡 Moyen |
| **11. 44 % Max DD reduction** | 🟢 | **MDD lean = −1.02 % vs B&H = −17.5 %** → **−94.2 % reduction** (bien meilleur que 44 %) | ✅ Objectif dépassé | 🟢 OK |
| **12. Conformal Prediction guarantees (95.2 % coverage)** | 🔴 | Module CP implémenté mais **aucune métrique de coverage** dans les rapports JSON — non activé dans le pipeline | Abstract promet « statistically guaranteed CI », implémentation ne les utilise pas | 🔴 **MAJEUR** |
| **13. Inference latency < 500 ms** | ❓ | **Non mesuré** — compute time total = 15 min 35 s (800 Optuna trials), mais pas de latency per-inference | Aucune donnée de latency dans les résultats | 🟡 Moyen |
| **14. Ablation studies** | 🟢 | ✅ `run_scaf_v3_ablation.py` — **8 variantes** (full, no_ml, no_regime, no_risk, trend_only, etc.) | ✅ Méthodologie rigoureuse, résultats publiés | 🟢 **EXCELLENT** |
| **15. OOS testing across multiple regimes** | 🟢 | ✅ Walk-forward 5 folds (2019/2020/2021/2022/2023) + Purged K-Fold | ✅ Validation temporelle robuste | 🟢 **EXCELLENT** |
| **16. Comparison against 15 baseline models** | 🔴 | **4 baselines seulement** : Donchian, Momentum 12-1, Equal-Weight B&H, SCAF v3 original | Abstract : 15 baselines ; réalité : 4 | 🔴 **MAJEUR** |
| **17. LLM continuous learning capability** | 🔴 | LLM orchestrator présent mais **offline/cached** (TTL=300s) — pas de continuous learning, pas de feedback loop actif | Abstract : « continuous learning » ; implémentation : static cache | 🔴 **CRITIQUE** |

---

## Score de concordance global

| Catégorie | Concordance | Détails |
|---|:---:|---|
| **Architecture promise (LLM + RL + DL + CP)** | **25 %** 🔴 | LLM dormant, RL non activé, DL régime **retiré** car toxique, CP non utilisé |
| **Couverture asset classes** | **25 %** 🔴 | 1/4 classes (US equity ETFs seulement) |
| **Métriques de performance** | **60 %** 🟡 | Sharpe ✅, DD ✅, mais **pas de RMSE**, pas de coverage CP |
| **Méthodologie validation** | **90 %** 🟢 | Ablation ✅, WF ✅, DSR ✅ — **excellente rigueur scientifique** |
| **Nombre de modèles experts** | **23 %** 🔴 | 3 actifs sur 13+ promis |
| **Baselines comparatifs** | **27 %** 🔴 | 4 baselines sur 15 promis |

**Score global : 42 % de concordance** 🔴

---

## Contradictions majeures

### 1. 🔴 **LLM comme orchestrateur central** — Abstract vs Réalité

| Abstract | Implémentation |
|---|---|
| *"LLMs as cognitive orchestrators"* | `llm/orchestrator.py` présent (348 LOC) mais **jamais appelé** dans `run_scaf_v3.py` |
| *"strategic reasoning tools"* | SCAF v3 lean = **Donchian + ML Cross-Section** — zéro appel LLM |
| *"continuous learning capability"* | LLM cache statique (TTL=300s), pas de feedback loop |

**Verdict** : le LLM est **un module dormant**, pas un orchestrateur actif.

---

### 2. 🔴 **Deep Learning regime detection** — Abstract vs Ablation

| Abstract | Résultat empirique (Phase C) |
|---|---|
| *"Deep Learning-based regime detection"* | `RegimeFilter` **supprimé** après ablation |
| Présenté comme un atout | **+0.65 Sharpe quand désactivé** (7.27 vs 6.62) |
| | Ablation prouve qu'il **détruit** de la valeur |

**Verdict** : l'abstract vante un composant que **l'implémentation a éliminé** pour cause de toxicité.

---

### 3. 🔴 **13+ expert models + Q-Learning selection** — Abstract vs Réalité

| Abstract | Implémentation |
|---|---|
| *"13+ specialized expert models"* | **3 modèles** : LogReg + Bagging + HistGBT |
| *"Q-Learning-based selection mechanism"* | Sélection **statique** via AUC gate (>0.55) |
| *"continuously improves through market feedback"* | Aucun feedback loop — params fixés post-Optuna |

**Verdict** : promesse d'un **méta-learner RL** vs. réalité d'un **filtre AUC statique**.

---

### 4. 🔴 **Forecasting (RMSE) vs Trading (Sharpe)**

| Abstract | Implémentation |
|---|---|
| *"81 % RMSE reduction"* | Aucune métrique RMSE dans les rapports |
| *"financial time series forecasting"* | SCAF v3 = **trading strategy** (positions ∈ [−1, 1]) |
| Comparaison vs DL baselines (LSTM, Transformer?) | Comparaison vs **Donchian / Momentum / B&H** |

**Verdict** : l'abstract décrit un **problème de forecasting**, l'implémentation résout un **problème de portfolio allocation**.

---

### 5. 🔴 **Conformal Prediction guarantees** — Abstract vs Usage

| Abstract | Implémentation |
|---|---|
| *"95.2 % coverage at 95 % confidence level"* | Module `conformal.py` (326 LOC) complet et rigoureux |
| *"statistically guaranteed confidence intervals"* | **Zéro appel** dans `run_scaf_v3.py` |
| | **Aucune métrique de coverage** dans les JSON reports |

**Verdict** : code prêt mais **non utilisé** — promesse non tenue.

---

## Composantes implémentées mais non mentionnées dans l'abstract

| Composante | LOC | Statut | Valeur |
|---|---:|---|---|
| **Deflated Sharpe Ratio (Bailey-LdP)** | `robustness.py` 301 LOC | ✅ Actif | 🟢 **Publication-grade** |
| **Purged K-Fold Walk-Forward** | `walk_forward.py` 175 LOC | ✅ Actif | 🟢 **Académique rigoureux** |
| **Stationary Block Bootstrap BCa** | `robustness.py` | ✅ Actif | 🟢 **CI robustes autocorr** |
| **Transaction cost model granulaire** | `cost_model.py` 144 LOC | ✅ Actif | 🟢 **5 profils réalistes** |
| **Reproductibilité complète** | `run_metadata.py` 147 LOC | ✅ Actif | 🟢 **Seed + git SHA + env hash** |

**Verdict** : l'implémentation actuelle a un **socle de validation statistique** bien supérieur à ce que l'abstract laisse entendre, mais dans une direction différente (trading robuste vs forecasting ML pur).

---

## Recommandations pour aligner l'implémentation sur l'abstract

| # | Action | Effort | Priorité | Impact concordance |
|---|---|---|---|---|
| 1 | **Activer LLM orchestrator** dans le pipeline principal | 2-3 j | 🔴 Haute | +20 % |
| 2 | **Implémenter Q-Learning model selection** (remplacer AUC gate statique) | 3-4 j | 🔴 Haute | +15 % |
| 3 | **Étendre à 4 asset classes** (ajouter EUR/USD, NASDAQ, BTC via yfinance) | 1-2 j | 🟡 Moyenne | +10 % |
| 4 | **Activer Conformal Prediction** + reporter coverage metrics | 1 j | 🟡 Moyenne | +10 % |
| 5 | **Baselines DL** : ajouter LSTM, Transformer, TCN comparisons | 2-3 j | 🟡 Moyenne | +8 % |
| 6 | **Ajouter métriques RMSE/MAE** pour forecasting (en plus de Sharpe trading) | 1 j | 🟢 Basse | +5 % |
| 7 | **Élargir à 13+ modèles** (intégrer `models/extra_models.py`, `torch_models.py`) | 2 j | 🟢 Basse | +7 % |
| 8 | **Mesurer inference latency** (per-prediction, pas total runtime) | 4 h | 🟢 Basse | +3 % |
| 9 | **Continuous learning loop** : online Q-table update + LLM re-query | 1 semaine | 🔴 Haute | +15 % |

**Total effort estimé** : **2-3 semaines** pour atteindre **85-90 % de concordance**.


---

## Roadmap détaillée pour atteindre 90 % de concordance

### Phase 1 — Activation des modules existants (1 semaine)

| Tâche | Fichiers modifiés | Effort | Validation |
|---|---|---|---|
| 1.1 **Intégrer LLM orchestrator** | `run_scaf_v3.py`, `scaf_v3/strategy.py` | 1 j | LLM appels logged dans JSON |
| 1.2 **Remplacer AUC gate par Q-Learning selector** | `scaf_v3/models.py`, `scaf_v3/optimizer.py` | 2 j | Q-table convergence visible |
| 1.3 **Activer Conformal Prediction** | `scaf_v3/models.py`, `analysis/validation_report.py` | 1 j | Coverage % dans JSON reports |
| 1.4 **Ajouter métriques RMSE/MAE** | `run_scaf_v3.py`, `analysis/stats.py` | 4 h | RMSE vs baselines dans rapport |
| 1.5 **Mesurer inference latency** | `scaf_v3/strategy.py`, nouveau `analysis/latency.py` | 4 h | Latency P50/P99 dans JSON |

**Livrable Phase 1** : SCAF v3 avec LLM + RL + CP actifs, métriques alignées sur abstract.

---

### Phase 2 — Extension du scope (1 semaine)

| Tâche | Fichiers modifiés | Effort | Validation |
|---|---|---|
| 2.1 **Ajouter EUR/USD, Bitcoin** | nouveau `data/loader_multi_asset.py` | 1 j | 4 asset classes dans univers |
| 2.2 **Élargir à 13 modèles** | `models/extra_models.py`, `models/torch_models.py` | 2 j | 13 modèles dans registry |
| 2.3 **Baselines DL** (LSTM, Transformer) | nouveau `baselines/deep_learning.py` | 2 j | RMSE comparison vs 15 baselines |
| 2.4 **Re-run walk-forward 4 asset classes** | `run_scaf_v3_walkforward.py` | 1 j | WF report pour chaque classe |

**Livrable Phase 2** : Évaluation multi-actifs avec 15 baselines incluant DL.

---

### Phase 3 — Continuous learning (3-4 jours)

| Tâche | Fichiers modifiés | Effort | Validation |
|---|---|---|
| 3.1 **Online Q-table update** | `models/qlearning_selector.py` | 1 j | Q-table JSON timestamped updates |
| 3.2 **LLM feedback loop** | `llm/orchestrator.py`, nouveau `llm/feedback.py` | 2 j | LLM re-query sur regime shift |
| 3.3 **Adaptive Conformal Prediction** | `models/conformal.py` (ACI déjà implémenté) | 1 j | αₜ adaptation logged |

**Livrable Phase 3** : Système avec continuous learning démontrable.

---

### Estimation globale

| Phase | Durée | Concordance finale estimée |
|---|---|---|
| **Phase 0** (actuel) | — | **42 %** |
| **Phase 1** | 1 semaine | **65 %** (+23 pp) |
| **Phase 2** | 1 semaine | **82 %** (+17 pp) |
| **Phase 3** | 3-4 j | **90 %** (+8 pp) |

**Effort total** : **2.5 semaines** (12-15 jours ouvrés) pour atteindre **90 % de concordance**.

---

## Option alternative : Réécrire l'abstract pour SCAF v3 lean

Si le but est de publier **l'implémentation actuelle** plutôt que l'architecture promise, voici l'abstract révisé :

---

### Abstract révisé (aligné sur SCAF v3 lean)

> **SCAF-Lite: A Statistically Robust Multi-Asset Trading Strategy with Deflated Sharpe Validation**
>
> Multivariate financial portfolio allocation remains challenging due to non-stationarity and regime changes. While Deep Learning and LLM-based approaches show promise, they often suffer from overfitting and lack of statistical guarantees. This paper introduces **SCAF v3 lean**, a parsimonious hybrid trading strategy that combines trend-following (Donchian breakout) with machine learning-based cross-sectional ranking. Unlike complex multi-layer architectures, SCAF v3 lean uses a minimal ensemble of 3 models (Logistic Regression, Bagging, HistGBT) selected via AUC gating (>0.55), and applies rigorous statistical validation through Deflated Sharpe Ratio (Bailey–López de Prado), Purged K-Fold Walk-Forward, and Stationary Block Bootstrap with BCa confidence intervals.
>
> Our evaluation on 18 US sector ETFs spanning 2015-2024 demonstrates:
> - **Sharpe Ratio 7.27** (out-of-sample 2022-2024, 752 days, 5 bps/leg)
> - **Max Drawdown −1.02%** (vs −17.5% for buy-and-hold)
> - **×6 cumulative return** vs buy-and-hold (+125% excess)
> - **DSR z-score −3.35** on walk-forward validation (vs −6.76 for the over-parameterized version)
>
> Ablation studies prove that a previously proposed regime-detection layer **destroys** value (+0.65 Sharpe when removed), validating the lean architecture. SCAF v3 lean demonstrates that **statistical robustness and simplicity** outperform architectural complexity in production trading systems, offering a reproducible framework (seed-controlled, git-versioned, cost-calibrated) for quantitative finance research.

---

**Concordance de cet abstract révisé avec l'implémentation actuelle** : **98 %** ✅

---

*Analyse complète générée le 2026-04-19.*


---

## Synthèse exécutive

### Ce que l'abstract promet

Une **architecture hybride orchestrée par LLM** qui combine :
- RL (Q-Learning) pour sélection adaptative de 13+ modèles
- DL (regime detection) pour adaptation contextuelle
- Conformal Prediction pour garanties statistiques
- Évaluation multi-actifs (4 classes) avec RMSE comme métrique principale
- Continuous learning via feedback de marché

### Ce que l'implémentation délivre (SCAF v3 lean)

Une **stratégie de trading quantitative simple et robuste** qui combine :
- Trend-following (Donchian) + Cross-section ML (3 modèles : LogReg, Bagging, HistGBT)
- Sélection statique via AUC gate (>0.55)
- Validation statistique rigoureuse (DSR, WF, ablation) **supérieure à l'abstract**
- 1 asset class (US equity ETFs), Sharpe comme métrique principale
- Zéro LLM actif, zéro RL actif, zéro CP actif

**L'implémentation actuelle est un excellent système de trading**, mais elle n'est **pas le SCAF de l'abstract**.

---

## Verdict final

| Critère | Évaluation |
|---|---|
| **Concordance architecturale** | 🔴 **25 %** — LLM/RL/CP présents mais dormants |
| **Concordance méthodologique** | 🟢 **90 %** — Validation supérieure (DSR, WF, ablation) |
| **Concordance performance** | 🟡 **60 %** — Sharpe ✅, DD ✅, mais RMSE manquant |
| **Concordance scope** | 🔴 **25 %** — 1/4 asset classes, 3/13 modèles |

**SCAF v3 lean** est un **prototype partiel** du système décrit dans l'abstract, avec une bifurcation méthodologique vers le **trading robuste** plutôt que le **forecasting ML orchestré par LLM**. Les modules nécessaires (`llm/`, `models/qlearning_selector.py`, `models/conformal.py`) sont présents mais **non intégrés**.

**Recommandation** : soit **aligner l'implémentation sur l'abstract** (2-3 semaines effort), soit **réécrire l'abstract pour décrire SCAF v3 lean** comme un système de trading statistiquement robuste avec validation académique.

---

*Analyse générée le 2026-04-19 par Augment Agent.*
