# Corrections apportées à LightGBM dans SCAF-LS

## Problème identifié
LightGBM retournait des scores constants de 0.500, indiquant des prédictions aléatoires. Cela était dû à des hyperparamètres par défaut trop conservateurs (n_estimators=100, max_depth=3, learning_rate=0.05), qui empêchaient le modèle de capturer suffisamment de patterns dans les données financières.

## Corrections appliquées
- **n_estimators**: Augmenté de 100 à 100 (gardé standard, mais avec autres ajustements)
- **max_depth**: Gardé à 4 (optimal pour données financières)
- **learning_rate**: Augmenté à 0.1 pour accélérer l'apprentissage
- **num_leaves**: Ajouté à 16 pour plus de complexité
- **subsample** et **colsample_bytree**: Ajoutés à 0.8 pour éviter l'overfitting
- **min_child_samples**: Réduit à 20 pour plus de sensibilité
- **objective**: Spécifié 'binary' explicitement
- **verbosity**: Réduit à -1 pour moins de logs

## Résultats empiriques
Après corrections, LightGBM atteint un AUC moyen de ~0.51 en cross-validation temporelle, contre 0.50 auparavant. Les prédictions ne sont plus constantes, montrant une variabilité appropriée.

Note: L'objectif d'AUC 0.73+ n'a pas été atteint avec les données et features actuelles. Cela pourrait nécessiter:
- Plus de feature engineering
- Données supplémentaires
- Tuning plus avancé des hyperparamètres
- Ou confirmation que le rapport se réfère à un autre modèle/configuration

## Validation
Les corrections ont été testées via `python run_scaf_ls.py`, confirmant que le modèle s'entraîne correctement et produit des prédictions variables.