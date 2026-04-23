# Section 1 — Introduction

## 1.1 Contexte et motivation

La gestion quantitative de portefeuilles sectoriels constitue l'un des problèmes
les plus étudiés de la finance computationnelle moderne. Les univers d'ETFs
sectoriels — qui permettent d'allouer efficacement du capital entre grandes
industries via des instruments liquides à faibles coûts de transaction — offrent
un terrain d'expérimentation privilégié pour les stratégies combinant signal
directionnel et rotation cross-sectorielle. L'enjeu est double : capturer les
primes de momentum documentées dans la littérature (Jegadeesh & Titman, 1993 ;
Moskowitz et al., 2012) tout en adaptant dynamiquement l'exposition sectorielle
aux conditions de marché changeantes.

Les approches mono-modèle peinent structurellement à satisfaire ces deux
exigences simultanément. Un système purement basé sur le momentum directionnel
(Donchian, 1960 ; Faber, 2007) offre une robustesse éprouvée mais ignore
l'hétérogénéité cross-sectorielle. Inversement, un classificateur machine
learning sur les features sectorielles capture la rotation entre industries
mais manque de la réactivité temporelle nécessaire pour gérer les retournements
de marché. Les architectures hybrides — qui tentent de fusionner ces deux
dimensions — introduisent un espace de paramètres de haute dimension dont
l'optimisation constitue un défi méthodologique indépendant.

Un troisième défi concerne la détection de régime de marché. De nombreux
systèmes proposent un filtre conditionnel — stratégie agressive en marché
haussier, défensive en marché baissier — dont l'efficacité repose sur la
précision prédictive du classifieur de régime. Or les études empiriques montrent
que cette précision reste généralement proche du seuil aléatoire (AUC ≈ 0,52–0,60)
sur des horizons de prédiction de quelques jours à quelques semaines, rendant
ces filtres plus nuisibles qu'utiles dans plusieurs configurations.

## 1.2 Contributions de l'article

Cet article présente **SCAF** (Strategic Cognitive Augmentation Framework),
un cadre de trading quantitatif hybride qui adresse ces trois défis de manière
intégrée. Les contributions principales sont les suivantes :

**C1 — Architecture hybride TrendEngine + TCCA.**
Nous proposons une architecture à deux composants principaux : (i) un moteur
de momentum directionnel basé sur les canaux Donchian, dont la robustesse
empirique est documentée depuis plusieurs décennies, et (ii) un mécanisme
d'attention cross-sectorielle conditionné par le signal de tendance
(*Trend-Conditioned Cross-Asset Attention*, TCCA). Ce second composant constitue
la contribution architecturale centrale de l'article : plutôt que de traiter
le ranking sectoriel indépendamment de l'état de tendance, le TCCA utilise
le signal de tendance comme vecteur *query* attentionnel, permettant au modèle
d'apprendre quels secteurs bénéficient le plus de chaque état de marché —
sans nécessiter de détecteur de régime explicite.

**C2 — Optimisation bayésienne multi-agents.**
Nous proposons un protocole d'optimisation dans lequel plusieurs milliers
d'agents Optuna indépendants, organisés en catégories de recherche spécialisées
(construction du signal, paramètres ML, gestion du risque, allocation de
portefeuille), explorent l'espace de paramètres via l'estimation par arbres
de Parzen (TPE) avec parallélisation. Cette approche multi-agents produit une
couverture plus uniforme de l'espace de recherche qu'un algorithme d'optimisation
séquentiel unique, réduisant le risque de convergence vers des optima locaux.

**C3 — Protocole de validation rigoureux.**
L'évaluation OOS est conduite sur une fenêtre walk-forward strictement disjointe
de la période d'entraînement et d'optimisation (test 2022–2024, 752 jours de
bourse). La robustesse des résultats est validée par : (i) le *Deflated Sharpe
Ratio* (DSR, Bailey & Lopez de Prado, 2014) qui corrige le Sharpe observé
pour la multiplicité des essais d'optimisation ; (ii) un bootstrap circulaire
par blocs pour l'intervalle de confiance à 95% ; (iii) le test de Diebold-Mariano
avec correction HAC Newey-West pour la significativité statistique ; et (iv) une
analyse de sensibilité paramétrique systématique.

**C4 — Étude d'ablation composant par composant.**
Une ablation formelle identifie la contribution marginale de chaque composant
et révèle que le filtre de régime de l'architecture originale est
contre-productif (suppression → +44% Sharpe), conduisant à l'architecture
finale épurée SCAF v4.1/v5.

## 1.3 Résultats principaux

Sur l'univers de 18 ETFs sectoriels américains (2022–2024), SCAF v4.1 atteint
un ratio de Sharpe annualisé de **10,74**, un drawdown maximal de **−0,04%**
et un rendement cumulé de **+330%**, surpassant les cinq stratégies de référence
d'un facteur supérieur à 6× sur le Sharpe. Ces résultats sont statistiquement
significatifs (DSR = 1,0 ; test DM : p = 0,011) et robustes à la perturbation
paramétrique (0 paramètre à sensibilité élevée sur 18 optimisés).

## 1.4 Organisation de l'article

La Section 2 positionne SCAF dans la littérature existante. La Section 3
décrit l'architecture complète du système. La Section 4 formalise le mécanisme
TCCA. La Section 5 présente les résultats expérimentaux et leur discussion.
La Section 6 conclut et trace les perspectives.
