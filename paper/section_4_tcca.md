# Section 4 — Trend-Conditioned Cross-Asset Attention (TCCA)

## 4.1 Motivation formelle

Soit $\mathcal{U} = \{1, \ldots, N\}$ l'univers de $N$ actifs sectoriels,
$\mathbf{f}_t^{(i)} \in \mathbb{R}^F$ le vecteur de features de l'actif $i$
à la date $t$, et $\tau_t \in [-1, 1]$ le signal de tendance calculé par
le TrendEngine sur le benchmark.

L'objectif du ranking cross-sectionnel est d'estimer une fonction de score :

$$s_t^{(i)} = g\!\left(\mathbf{f}_t^{(i)}\right)$$

telle que $s_t^{(i)} > s_t^{(j)}$ implique $\mathbb{E}[r_{t+h}^{(i)}] > \mathbb{E}[r_{t+h}^{(j)}]$,
où $r_{t+h}^{(i)}$ est le rendement à horizon $h$ de l'actif $i$.

Les approches existantes traitent $g$ comme une fonction symétrique : tous
les actifs contribuent indépendamment au score. Cette formulation ignore
l'information directionnelle disponible via $\tau_t$.

TCCA reformule le problème comme un **scoring conditionnel** :

$$s_t^{(i)} = g\!\left(\mathbf{f}_t^{(i)},\; \tau_t\right)$$

Le score de l'actif $i$ dépend non seulement de ses propres features, mais
aussi de l'état de tendance courant du marché. L'intuition économique est claire :
les secteurs cycliques (XLK, XLE) surperforment conditionnellement en marché
haussier fort, tandis que les secteurs défensifs (XLV, XLP) sont préférables
en tendance négative. TCCA apprend cette dépendance conditionnelle end-to-end,
sans nécessiter d'annotation manuelle des régimes.

## 4.2 Architecture TCCA

### 4.2.1 Projection du signal de tendance (Query)

Le signal scalaire $\tau_t$ est projeté dans l'espace d'embedding via un
réseau à deux couches avec activation GELU :

$$\mathbf{q}_t = W_2 \cdot \text{GELU}(W_1 \cdot \tau_t + b_1) + b_2 \in \mathbb{R}^d$$

où $d$ est la dimension d'embedding (hyperparamètre). Ce vecteur $\mathbf{q}_t$
constitue le **query** unique qui interroge l'ensemble des actifs — une
représentation apprise de l'état de tendance courant.

### 4.2.2 Projections des features sectorielles (Keys et Values)

Les features de chaque actif sont projetées linéairement :

$$\mathbf{k}_t^{(i)} = W_K \cdot \mathbf{f}_t^{(i)} \in \mathbb{R}^d$$
$$\mathbf{v}_t^{(i)} = W_V \cdot \mathbf{f}_t^{(i)} \in \mathbb{R}^d$$

$\mathbf{k}_t^{(i)}$ encode "ce que l'actif $i$ offre comme signal" et
$\mathbf{v}_t^{(i)}$ encode "ce que l'actif $i$ contient comme information".

### 4.2.3 Mécanisme d'attention multi-têtes

L'attention est calculée entre le query de tendance et les keys de tous les
actifs simultanément. Pour $H$ têtes d'attention de dimension $d_h = d/H$ :

$$\alpha_t^{(h,i)} = \frac{\exp\!\left(\mathbf{q}_t^{(h)} \cdot \mathbf{k}_t^{(h,i)} / \sqrt{d_h}\right)}
{\sum_{j=1}^{N} \exp\!\left(\mathbf{q}_t^{(h)} \cdot \mathbf{k}_t^{(h,j)} / \sqrt{d_h}\right)}$$

Les poids d'attention $\alpha_t^{(h,i)} \in [0,1]$ représentent l'importance
relative de l'actif $i$ étant donné l'état de tendance $\tau_t$ pour la tête $h$.

Le vecteur de contexte agrégé est :

$$\mathbf{c}_t = W_O \cdot \text{Concat}_{h=1}^H\left(\sum_{i=1}^N \alpha_t^{(h,i)} \mathbf{v}_t^{(h,i)}\right) \in \mathbb{R}^d$$

Ce vecteur $\mathbf{c}_t$ encode "quelle information sectorielle est pertinente
compte tenu de la tendance courante". Il est broadcasté sur tous les actifs.

### 4.2.4 Résidu, normalisation et scoring

Chaque actif reçoit la somme de sa propre projection value et du contexte
global conditionné par la tendance :

$$\mathbf{o}_t^{(i)} = \text{LayerNorm}\!\left(\mathbf{v}_t^{(i)} + \mathbf{c}_t\right)$$

Une couche FFN à deux niveaux avec activation GELU affine la représentation :

$$\tilde{\mathbf{o}}_t^{(i)} = \text{LayerNorm}\!\left(\mathbf{o}_t^{(i)} + \text{FFN}\!\left(\mathbf{o}_t^{(i)}\right)\right)$$

Le score de ranking final est obtenu par projection scalaire :

$$s_t^{(i)} = \mathbf{w}_s^\top \tilde{\mathbf{o}}_t^{(i)} \in \mathbb{R}$$

### 4.2.5 Complexité paramétrique

Le nombre total de paramètres est :

$$|\theta| = 2d + 2d^2 + 3dF + d^2 + 2d + 2d^2 + d \approx 5d^2 + 3dF + 3d$$

Pour $d = 32$, $F = 58$ : $|\theta| \approx 5 \times 1024 + 3 \times 1856 + 96 \approx 10{,}720$ paramètres.

Ce modèle ultra-compact (< 15 000 paramètres) minimise le risque d'overfitting
sur les 1 258 jours d'entraînement disponibles, tout en étant suffisamment
expressif pour capturer les dépendances conditionnelles non-linéaires entre
tendance de marché et rotation sectorielle.

## 4.3 Fonction de perte — Ranking pairwise

L'objectif d'entraînement est de maximiser la corrélation de rang entre
les scores prédits et les rendements futurs. Nous utilisons une perte de
ranking pairwise basée sur la BCE :

$$\mathcal{L}(\theta) = -\frac{1}{|B|} \sum_{(t,t') \in B} \sum_{i,j}
\mathbb{1}[r_{t+h}^{(i)} > r_{t+h}^{(j)}] \cdot
\log \sigma\!\left(s_t^{(i)} - s_t^{(j)}\right)$$

où $B$ est un mini-batch de dates, $\sigma$ la fonction sigmoïde, et
l'indicateur $\mathbb{1}[\cdot]$ fournit le label de ranking supervisé.

Cette formulation entraîne le modèle à produire $s_t^{(i)} > s_t^{(j)}$
chaque fois que l'actif $i$ surperforme $j$ sur l'horizon $h=5$ jours.
Le gradient est non nul pour toutes les paires $(i,j)$ avec ordres inverses,
ce qui fournit $O(N^2)$ signaux d'apprentissage par date — crucial pour
compenser la faible longueur de la série temporelle.

L'optimiseur est Adam (Kingma & Ba, 2015) avec weight decay $10^{-5}$ et
scheduler cosinus. Le gradient est clipé à norme 1 pour la stabilité.

## 4.4 Conversion scores → poids de portefeuille

Les scores bruts $s_t^{(i)}$ sont convertis en pseudo-probabilités via
la fonction sigmoïde :

$$p_t^{(i)} = \sigma(s_t^{(i)}) \in (0, 1)$$

Ces pseudo-probabilités sont passées à `CrossSectionPortfolio` qui applique
des seuils de long/short :

$$w_t^{(i)} = \begin{cases}
+w^+ & \text{si } p_t^{(i)} \geq \text{long\_thr} \text{ (top-4)} \\
-w^- & \text{si } p_t^{(i)} \leq \text{short\_thr} \text{ (bottom-4)} \\
0 & \text{sinon}
\end{cases}$$

avec $\sum_i (w_t^{(i)})^+ = 1$ et $\sum_i (w_t^{(i)})^- = -1$.

## 4.5 Interprétabilité par les poids d'attention

Un avantage important du TCCA est son interprétabilité inhérente.
Les poids d'attention $\bar{\alpha}_t^{(i)} = \frac{1}{H}\sum_h \alpha_t^{(h,i)}$
indiquent directement quelle importance le modèle accorde à chaque secteur
conditionnellement à l'état de tendance $\tau_t$.

L'analyse des poids moyens par tertile de $\tau_t$ révèle les patterns
appris :

| État de tendance | Secteurs favorisés | Secteurs défavorisés |
|---|---|---|
| Fort haussier (τ > 0.5) | XLK, XLC, XLY | XLP, XLU, TLT |
| Neutre (-0.2 < τ < 0.2) | Distribution uniforme | — |
| Baissier (τ < -0.5) | TLT, GLD, XLV | XLE, XLF, XLY |

Ces patterns sont économiquement cohérents avec la théorie de la rotation
sectorielle (Stovall, 1996) : les secteurs cycliques surperforment en expansion,
les défensifs et les actifs refuges en contraction. Le TCCA apprend ces
relations implicitement depuis les données, sans encodage manuel.

## 4.6 Comparaison avec les approches existantes

| Approche | Query | Keys/Values | Conditionnement |
|---|---|---|---|
| Self-Attention standard | Chaque actif s'interroge lui-même | Tous les actifs | Symétrique |
| Graph Attention (GAT) | Voisins dans le graphe | Actifs connectés | Topologique |
| Cross-Attention (texte↔image) | Modalité A | Modalité B | Bi-modal |
| **TCCA (ici)** | **Signal de tendance** | **Features sectorielles** | **Directionnel** |

La distinction fondamentale de TCCA est que le **query est déterministe et
exogène** (calculé par règle fixe sur SPY), tandis que les **keys/values sont
appris et endogènes** (paramètres du réseau). Cette asymétrie crée une
hiérarchie d'information claire : le signal rule-based *guide* l'apprentissage
ML plutôt que de le *remplacer* ou d'être *remplacé* par lui.

## 4.7 Intégration dans le protocole walk-forward

L'entraînement du TCCA suit strictement le protocole walk-forward :

1. **Phase IS (2015–2019, 1 258 jours)** : entraînement initial avec hyperparamètres explorés par Optuna
2. **Phase OPT (2020–2021, 505 jours)** : évaluation de chaque trial Optuna (Sharpe OPT = objectif)
3. **Phase finale** : ré-entraînement sur IS+OPT (1 763 jours) avec best_params, 60 epochs
4. **Phase OOS (2022–2024, 752 jours)** : évaluation finale, aucune adaptation du modèle

À aucun moment les données OOS n'influencent les paramètres du modèle,
garantissant l'absence de look-ahead bias.
