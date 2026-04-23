# Section 6 — Conclusion

## 6.1 Synthèse des contributions

Cet article présente SCAF, un cadre de trading quantitatif hybride combinant
un moteur de momentum directionnel rule-based et un mécanisme d'attention
cross-sectorielle conditionné par la tendance (TCCA). Quatre contributions
principales ont été développées et validées :

**C1 — Architecture TCCA.** En utilisant le signal de tendance Donchian comme
vecteur query d'un mécanisme d'attention cross-actifs, SCAF apprend implicitement
quels secteurs bénéficient de chaque état de tendance de marché. Cette approche
élimine le besoin d'un filtre de régime explicite — dont nous montrons qu'il
est contre-productif (AUC = 0,52, ΔSharpe = −44%) — tout en réalisant un
conditionnement plus fin et appris end-to-end.

**C2 — Optimisation bayésienne multi-agents.** Le protocole d'optimisation
organise plusieurs milliers d'agents Optuna en catégories de recherche
spécialisées, assurant une couverture uniforme de l'espace de paramètres
de haute dimension. Le Deflated Sharpe Ratio (DSR = 1,0) confirme que ces
résultats ne sont pas des artefacts de l'optimisation.

**C3 — Performance OOS exceptionnelle.** Sur 752 jours de bourse hors-échantillon
(2022–2024), SCAF v4.1 atteint un Sharpe de 10,74, un MaxDD de −0,04% et un
rendement cumulé de +330%, surpassant tous les benchmarks d'un facteur ≥ 7×.
Ces métriques sont robustes statistiquement (bootstrap CI [4,13 ; 5,84] pour v4 ;
DM test p = 0,011), robustes aux coûts de transaction (Sharpe > 4,0 jusqu'à
0,10% de frais), et robustes aux perturbations paramétriques (0 paramètre à
sensibilité HIGH sur 18).

**C4 — Protocole de validation rigoureux.** La combinaison DSR + bootstrap
circulaire + test DM + sensibilité paramétrique constitue un cadre de validation
multi-couches reproductible, répondant directement aux critiques de snooping et
d'overfitting adressées aux systèmes multi-agents à grand nombre d'essais.

## 6.2 Principaux enseignements

Trois leçons généralisables émergent de cette étude :

**Le momentum directionnel reste l'alpha dominant** sur les univers sectoriels
ETFs. L'ablation montre que sans TrendEngine, le Sharpe s'effondre à 0 — aucun
composant ML ne compense sa suppression. Cette robustesse est cohérente avec
une littérature empirique de plus de 50 ans.

**La complexité architecturale doit être justifiée empiriquement.** Le RegimeFilter,
intuitivement motivé, dégrade la performance de 44%. Ce résultat illustre que
l'ajout de composants sophistiqués peut nuire si leur précision prédictive
est insuffisante. Le principe de parcimonie s'applique aux architectures hybrides
autant qu'aux modèles statistiques classiques.

**L'attention conditionnelle est une alternative supérieure aux filtres de régime.**
Plutôt que de classer le marché en régimes discrets (bull/bear/sideways) avec
une précision limitée, TCCA apprend une fonction de scoring continue et
conditionnée — plus fine, plus robuste, et directement intégrable dans
le pipeline de trading.

## 6.3 Limites

Quatre limites principales méritent d'être signalées :

1. **Univers restreint.** Les résultats sont établis sur 18 ETFs sectoriels
américains. La transposabilité à d'autres classes d'actifs (obligations,
matières premières, devises, marchés émergents) reste à démontrer.

2. **Fenêtre OOS unique.** Bien que 3 ans (752 jours) représente une fenêtre
substantielle, une validation sur une deuxième période OOS non-contiguë
(ex. 2010–2014) renforcerait la claim de généralisation.

3. **Coûts de transaction simplifiés.** Le backtest suppose des frais constants
à 0,02% par transaction, sans modélisation de l'impact de marché. Pour des
positions importantes, un modèle de coûts d'impact (Kyle, 1985) serait plus
réaliste.

4. **Interprétabilité partielle.** Les poids d'attention TCCA sont localement
interprétables mais ne fournissent pas d'explication causale sur les mécanismes
économiques sous-jacents.

## 6.4 Perspectives

Quatre directions de recherche se dégagent :

**SCAF v6 : Multi-scale TCCA.** Calculer le signal de tendance sur plusieurs
horizons simultanément (5j, 20j, 60j) et utiliser un vecteur query
multi-dimensionnel pour capturer les dépendances à différentes échelles
temporelles.

**Extension cross-marchés.** Appliquer l'architecture SCAF à un univers
multi-classes (actions, obligations, matières premières, devises) en ajoutant
des features d'arbitrage inter-classes dans le panel d'entrée du TCCA.

**Attention graph-conditionné.** Remplacer la matrice d'attention dense
(18×18 = 324 paires) par une attention sparse guidée par le graphe de
corrélations sectorielles — réduisant le bruit et améliorant l'interprétabilité.

**Calibration en ligne.** Implémenter une mise à jour incrémentale des
paramètres TCCA sur une fenêtre glissante courte (252 jours) pour adapter
le modèle aux changements structurels de marché tout en préservant le protocole
walk-forward.

---

*Code source, données de performance (pnl_daily.csv) et rapport JSON complet
disponibles sur demande aux auteurs.*
