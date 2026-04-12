# Données CSV pré-téléchargées

Ce répertoire contient les données de marché réelles téléchargées localement,
permettant d'exécuter SCAF-LS **sans accès internet** (sandbox CI, Codespaces, etc.).

## Fichiers attendus

| Fichier          | Ticker  | Description             |
|------------------|---------|-------------------------|
| `GSPC.csv`       | ^GSPC   | S&P 500 (actif principal) |
| `VIX.csv`        | ^VIX    | CBOE Volatility Index   |
| `TNX.csv`        | ^TNX    | US 10-Year Treasury     |
| `IRX.csv`        | ^IRX    | US 13-Week T-Bill       |
| `GOLD.csv`       | GC=F    | Gold Futures            |
| `OIL.csv`        | CL=F    | Crude Oil Futures       |

## Comment générer ces fichiers

Exécutez le script fourni **une seule fois** sur votre machine locale (avec accès internet) :

```bash
cd 07-04-2026
python data/download_data.py
```

Le script télécharge les données 2018-2024 via `yfinance` et les sauvegarde dans ce répertoire.
Vous pouvez ensuite versionner ces CSV ou les copier dans un environnement sans internet.
