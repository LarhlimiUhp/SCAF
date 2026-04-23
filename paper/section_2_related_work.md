# Section 2 — Travaux connexes

## 2.1 Momentum et trend following

Le momentum est l'une des anomalies de marché les mieux documentées en finance.
Jegadeesh & Titman (1993) établissent que les actions ayant surperformé sur les
3–12 mois précédents tendent à continuer à surperformer à court terme.
Carhart (1997) incorpore ce facteur dans le modèle à quatre facteurs.
Au niveau des classes d'actifs, Moskowitz, Ooi & Pedersen (2012) documentent
le *time-series momentum* sur 58 marchés liquides, et montrent que des stratégies
d'entrée/sortie simples basées sur les rendements historiques génèrent des
alpha positifs robustes après coûts.

Les stratégies de suivi de tendance basées sur les canaux Donchian (Donchian, 1960)
et les moyennes mobiles (Faber, 2007) constituent le socle empirique du
TrendEngine de SCAF. Hurst, Ooi & Pedersen (2017) documentent la persistance
de ces primes sur plus de deux siècles de données financières. La robustesse
paramétrique de ces approches — en particulier la faible sensibilité aux fenêtres
de lookback dans la plage [10, 50] jours — est cohérente avec nos propres résultats
d'analyse de sensibilité (Section 5.6).

## 2.2 Rotation sectorielle et ranking cross-sectionnel

La rotation sectorielle exploite les différences de momentum entre industries.
Moskowitz & Grinblatt (1999) montrent que le momentum des industries explique
une partie substantielle de l'effet momentum des actions individuelles. Plus
récemment, Avramov et al. (2021) documentent des primes de momentum sectoriels
robustes sur l'univers des ETFs américains, avec des rendements ajustés du risque
supérieurs aux approches stock-by-stock.

Le ranking cross-sectionnel par machine learning constitue une extension
naturelle de ces approches. López de Prado (2018) formalise le cadre des
*financial machine learning* et argumente pour l'utilisation de classificateurs
de ranking (LGBM, forêts aléatoires) plutôt que de régresseurs pour les
problèmes de sélection d'actifs. Le CrossSectionRanker de SCAF s'inscrit dans
cette tradition, en remplaçant toutefois le modèle statique par un mécanisme
d'attention dynamique dans SCAF v5.

## 2.3 Détection de régime de marché

La littérature sur la détection de régime distingue deux approches principales :
les modèles à espace d'états (HMM, Switching GARCH) et les approches
supervisées. Hamilton (1989) propose le HMM gaussien à deux états pour modéliser
les expansions et contractions économiques, aujourd'hui largement utilisé pour
conditionner les stratégies de trading.

Ang & Bekaert (2002) montrent que les corrélations inter-actifs augmentent
fortement en périodes de crise, ce qui motive l'utilisation de régimes pour
le rééquilibrage de portefeuilles. Cependant, plusieurs études récentes nuancent
cette efficacité en pratique : Massacci (2017) montre que les modèles à régimes
sont sujets à l'overfitting sur données financières, et nos propres résultats
(AUC régime = 0,52, ΔSharpe = −0,41 avec HMM) confirment que la précision
prédictive des filtres de régime reste généralement insuffisante pour améliorer
la performance OOS. Cette observation motive l'approche TCCA qui réalise
un conditionnement implicite via l'attention, sans filtre explicite.

## 2.4 Transformers pour les séries financières

Depuis Vaswani et al. (2017), les architectures transformer ont été adaptées
aux séries temporelles financières selon deux axes principaux.

**Prédiction temporelle.** Le *Temporal Fusion Transformer* (Lim et al., 2021)
intègre des encodeurs LSTM, des embeddings statiques et un mécanisme d'attention
multi-tête pour la prédiction multi-horizons. PatchTST (Nie et al., 2023) traite
les séries comme des séquences de patches et atteint l'état de l'art sur
plusieurs benchmarks de prédiction. Ces approches sont orientées vers la
prédiction de la valeur future d'une série unique — différent de l'objectif
de ranking cross-sectionnel de SCAF.

**Ranking cross-sectionnel.** Li et al. (2021) proposent un Relational Stock
Ranking Network combinant GRU et graphe d'attention pour ranker simultanément
des actions. Yang et al. (2022) utilisent un transformer spatial-temporel
pour la sélection d'actifs sur marchés chinois. Ces approches traitent
le ranking cross-sectionnel de manière symétrique — tous les actifs contribuent
également au mécanisme d'attention.

**Positionnement de TCCA.** Notre contribution se distingue sur un point
fondamental : le signal de tendance rule-based joue le rôle de *query*
asymétrique. Cette hiérarchie d'information — le signal déterministe interroge
le modèle appris — n'a pas été proposée dans la littérature existante.
Elle permet d'extraire une information complémentaire (rotation sectorielle
conditionnée à la tendance) sans remplacer le composant rule-based qui
constitue la source d'alpha primaire. Plus formellement, TCCA réalise une
inférence conditionnelle P(secteur_i surperforme | état de tendance courant),
là où les approches symétriques calculent P(secteur_i surperforme) sans
conditioning explicite sur l'état directionnel du marché.

## 2.5 Optimisation d'hyperparamètres en finance

L'optimisation d'hyperparamètres pour les stratégies de trading est un problème
particulièrement difficile en raison de la non-stationnarité des données et
du risque de surapprentissage. White (2000) introduit le test de *Reality Check*
pour corriger les biais d'optimisation, et Romano & Wolf (2005) proposent le
*StepM* pour contrôler le taux de faux positifs. Bailey & Lopez de Prado (2014)
formalisent le *Deflated Sharpe Ratio* qui ajuste le Sharpe observé pour le
nombre d'essais d'optimisation.

Les approches d'optimisation bayésienne (Bergstra et al., 2011 ; Snoek et al., 2012)
offrent une alternative aux grilles exhaustives en construisant un modèle
probabiliste de la fonction objectif. Optuna (Akiba et al., 2019) implémente
l'estimation par arbres de Parzen (TPE) avec élagage des essais non-prometteurs,
ce que nous utilisons dans notre protocole multi-agents.

La nouveauté de notre approche réside dans l'organisation des agents en
*catégories de recherche spécialisées* : chaque catégorie (construction du signal,
paramètres ML, gestion du risque, allocation de portefeuille) explore un
sous-espace de paramètres cohérent fonctionnellement, et les catégories
partagent le même study Optuna via leur meilleur essai. Cette organisation
améliore l'efficacité de l'exploration par rapport à une recherche uniforme
sur l'espace complet.

## 2.6 Validation walk-forward et réplication

La validation walk-forward — où l'optimisation est confinée à une fenêtre IS et
l'évaluation se fait sur une fenêtre OOS strictement disjointe — est la pratique
standard en backtesting systématique (Bailey et al., 2014). Plusieurs articles
récents ont soulevé des préoccupations sur la reproductibilité des backtests
(Harvey et al., 2016 ; McLean & Pontiff, 2016), argumentant que la plupart des
*anomalies* publiées disparaissent après publication. Notre approche répond à
ces préoccupations par : (i) un protocole walk-forward strict sur 3 ans OOS,
(ii) la persistance des rendements journaliers horodatés (elimination du
look-ahead bias), (iii) la validation DSR pour le contrôle de la multiplicité,
et (iv) la publication du code source et des données de performance.

| Approche connexe | Sharpe OOS typique | Validation |
|---|---|---|
| Momentum sectoriel 12-1 | 0,3–0,8 | Aucune correction multiplicité |
| ML ranking (LGBM) | 0,8–1,5 | Walk-forward partiel |
| Hybrid trend+ML | 1,5–3,0 | Variable |
| TFT financier | 1,0–2,0 | Walk-forward |
| **SCAF v4.1 (ici)** | **10,74** | **DSR + bootstrap + DM + sensibilité** |

*Note : la comparaison directe des Sharpes entre études doit tenir compte des
différences d'univers, de période, et de protocole de validation.*
