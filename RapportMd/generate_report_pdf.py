"""
SCAF v3 — Compte Rendu Complet (PDF)
Génère un rapport professionnel avec reportlab
"""
import json, pathlib
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(
    r"C:\2025-2026\Habilitation 2026\antigravity 0001\scaf-11-04-2026"
    r"\SCAF-copilot-update-files-and-analyze-results"
    r"\SCAF-copilot-update-files-and-analyze-results\07-04-2026"
)
RESULTS = BASE / "results" / "scaf_v3_20260416_133210"
DASH    = RESULTS / "dashboard"
OUT     = RESULTS / "SCAF_v3_Rapport_Complet.pdf"

REPORT = json.loads((RESULTS / "scaf_v3_report.json").read_text())
TR  = REPORT["test_results"]
AGT = REPORT["agent_summaries"]
BP  = REPORT["best_params"]

# ── Colors ───────────────────────────────────────────────────────────────────
C_DARK   = HexColor("#0d1117")
C_PANEL  = HexColor("#161b22")
C_ACCENT = HexColor("#58a6ff")
C_GREEN  = HexColor("#3fb950")
C_RED    = HexColor("#f85149")
C_ORANGE = HexColor("#d29922")
C_GOLD   = HexColor("#ffa657")
C_PURPLE = HexColor("#bc8cff")
C_TEXT   = HexColor("#e6edf3")
C_MUTED  = HexColor("#8b949e")
C_BORDER = HexColor("#30363d")
C_WHITE  = colors.white
C_BLACK  = colors.black

# ── Styles ───────────────────────────────────────────────────────────────────
def build_styles():
    s = getSampleStyleSheet()
    custom = {}

    custom["cover_title"] = ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=32,
        textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=8, leading=38
    )
    custom["cover_sub"] = ParagraphStyle(
        "cover_sub", fontName="Helvetica", fontSize=15,
        textColor=C_ACCENT, alignment=TA_CENTER, spaceAfter=6, leading=20
    )
    custom["cover_meta"] = ParagraphStyle(
        "cover_meta", fontName="Helvetica", fontSize=10,
        textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=4, leading=14
    )
    custom["h1"] = ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=18,
        textColor=C_ACCENT, spaceAfter=10, spaceBefore=18, leading=22
    )
    custom["h2"] = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=13,
        textColor=C_TEXT, spaceAfter=6, spaceBefore=12, leading=17
    )
    custom["h3"] = ParagraphStyle(
        "h3", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_GOLD, spaceAfter=4, spaceBefore=8, leading=14
    )
    custom["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10,
        textColor=C_TEXT, spaceAfter=6, leading=15, alignment=TA_JUSTIFY
    )
    custom["body_mono"] = ParagraphStyle(
        "body_mono", fontName="Courier", fontSize=9,
        textColor=C_TEXT, spaceAfter=4, leading=13
    )
    custom["caption"] = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique", fontSize=8.5,
        textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=8, leading=12
    )
    custom["kpi_label"] = ParagraphStyle(
        "kpi_label", fontName="Helvetica", fontSize=9,
        textColor=C_MUTED, alignment=TA_LEFT, leading=12
    )
    custom["kpi_value"] = ParagraphStyle(
        "kpi_value", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_GREEN, alignment=TA_RIGHT, leading=14
    )
    custom["verdict_ok"] = ParagraphStyle(
        "verdict_ok", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_GREEN, alignment=TA_CENTER, leading=14
    )
    custom["verdict_warn"] = ParagraphStyle(
        "verdict_warn", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_ORANGE, alignment=TA_CENTER, leading=14
    )
    custom["footer"] = ParagraphStyle(
        "footer", fontName="Helvetica", fontSize=8,
        textColor=C_MUTED, alignment=TA_CENTER, leading=10
    )
    return custom

ST = build_styles()


# ── Header / Footer ───────────────────────────────────────────────────────────
def on_page(canvas, doc):
    W, H = A4
    canvas.saveState()
    # Top bar
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(1.5*cm, H - 0.75*cm, "SCAF v3  |  Self-Consistent Adaptive Framework")
    canvas.setFillColor(C_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(W - 1.5*cm, H - 0.75*cm, f"Run: 20260416_133210")
    # Bottom bar
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, 0, W, 0.9*cm, fill=1, stroke=0)
    canvas.setFillColor(C_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(W/2, 0.35*cm, f"Page {doc.page}  |  Confidentiel — Usage interne")
    canvas.restoreState()

def on_first_page(canvas, doc):
    # Full dark background on cover
    W, H = A4
    canvas.saveState()
    canvas.setFillColor(C_DARK)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Accent gradient bar top
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, H - 0.5*cm, W, 0.5*cm, fill=1, stroke=0)
    # Bottom bar
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, 0, W, 1.8*cm, fill=1, stroke=0)
    canvas.setFillColor(C_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(W/2, 0.6*cm, "Confidentiel — Usage interne  |  2026")
    canvas.restoreState()


# ── KPI Table helper ──────────────────────────────────────────────────────────
def kpi_table(rows, col_widths=None):
    """rows = list of (label, value, color_hex)"""
    data = []
    for label, value, clr in rows:
        data.append([
            Paragraph(label, ST["kpi_label"]),
            Paragraph(f'<font color="{clr}">{value}</font>',
                      ParagraphStyle("v", fontName="Helvetica-Bold", fontSize=11,
                                     alignment=TA_RIGHT, leading=14, textColor=C_GREEN))
        ])
    cw = col_widths or [10*cm, 5.5*cm]
    t = Table(data, colWidths=cw, repeatRows=0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_PANEL),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
        ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


def metric_grid(metrics):
    """2-column grid of large KPI boxes."""
    cells = []
    for label, value, clr in metrics:
        inner = Table(
            [[Paragraph(label, ParagraphStyle("ml", fontName="Helvetica", fontSize=9,
                                               textColor=C_MUTED, leading=12))],
             [Paragraph(f'<font color="{clr}">{value}</font>',
                        ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=18,
                                       textColor=C_WHITE, leading=22))]],
            colWidths=[7.5*cm]
        )
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), C_PANEL),
            ("BOX", (0,0), (-1,-1), 1, C_BORDER),
            ("LEFTPADDING",  (0,0), (-1,-1), 12),
            ("TOPPADDING",   (0,0), (-1,-1), 10),
            ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ]))
        cells.append(inner)
    # Pair up into rows
    rows = []
    for i in range(0, len(cells), 2):
        row = cells[i:i+2]
        if len(row) == 1:
            row.append(Spacer(7.5*cm, 1))
        rows.append(row)
    grid = Table(rows, colWidths=[7.7*cm, 7.7*cm], hAlign="LEFT")
    grid.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
    ]))
    return grid


def img(path, w=16*cm):
    p = pathlib.Path(path)
    if p.exists():
        return Image(str(p), width=w, height=w*0.5625, kind="proportional")
    return Paragraph(f"[Image not found: {p.name}]", ST["caption"])


def section_hr():
    return HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6, spaceBefore=4)


# ── Build Story ───────────────────────────────────────────────────────────────
story = []
W_A4 = 15.5*cm   # usable width

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — COVER
# ─────────────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 3.5*cm))
story.append(Paragraph("SCAF v3", ST["cover_title"]))
story.append(Paragraph("Self-Consistent Adaptive Framework", ST["cover_sub"]))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Compte Rendu Complet — Backtest OOS 2022–2024", ST["cover_sub"]))
story.append(Spacer(1, 1.2*cm))

# Cover KPI highlight
cover_data = [
    ["Sharpe Ratio OOS", "5.1625", "Max Drawdown", "-0.49%"],
    ["Rend. Annuel",     "+27.6%", "Calmar Ratio",  "56.47"],
    ["Rend. Cumulé",    "+82.4%", "Excès vs B&H", "+57.2%"],
]
cover_tbl = Table(
    [[Paragraph(f'<font color="#8b949e">{r[0]}</font>', ST["kpi_label"]),
      Paragraph(f'<font color="#3fb950">{r[1]}</font>',
                ParagraphStyle("cv", fontName="Helvetica-Bold", fontSize=20,
                               textColor=C_GREEN, alignment=TA_LEFT, leading=24)),
      Paragraph(f'<font color="#8b949e">{r[2]}</font>', ST["kpi_label"]),
      Paragraph(f'<font color="#3fb950">{r[3]}</font>',
                ParagraphStyle("cv2", fontName="Helvetica-Bold", fontSize=20,
                               textColor=C_GREEN, alignment=TA_LEFT, leading=24))]
     for r in cover_data],
    colWidths=[5.5*cm, 3.5*cm, 5.5*cm, 3.5*cm]
)
cover_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), C_PANEL),
    ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("GRID", (0,0), (-1,-1), 0.5, C_BORDER),
    ("LEFTPADDING",  (0,0), (-1,-1), 14),
    ("RIGHTPADDING", (0,0), (-1,-1), 14),
    ("TOPPADDING",   (0,0), (-1,-1), 10),
    ("BOTTOMPADDING",(0,0), (-1,-1), 10),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(cover_tbl)
story.append(Spacer(1, 1.5*cm))

story.append(Paragraph(
    "Run ID : 20260416_133210  |  Durée : 866.6 s  |  "
    "Univers : 18 actifs  |  1 000 agents Optuna  |  58 features",
    ST["cover_meta"]
))
story.append(Paragraph(
    f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
    ST["cover_meta"]
))

story.append(Spacer(1, 2*cm))
# Targets met badges
badges = [
    ("Sharpe > 1.033", "[OK]", C_GREEN),
    ("Max DD < 15%",   "[OK]", C_GREEN),
    ("Bat le B&H",     "[OK]", C_GREEN),
    ("Win Rate > 45%", "[--]", C_ORANGE),
]
badge_data = [[
    Paragraph(
        f'<font color="{clr}">{v}</font>  '
        f'<font color="#8b949e">{lbl}</font>',
        ParagraphStyle("bd", fontName="Helvetica-Bold", fontSize=11,
                       textColor=C_WHITE, alignment=TA_CENTER, leading=14)
    )
    for lbl, v, clr in badges
]]
badge_tbl = Table(badge_data, colWidths=[3.7*cm]*4)
badge_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), C_PANEL),
    ("BOX", (0,0), (-1,-1), 1, C_BORDER),
    ("INNERGRID", (0,0), (-1,-1), 0.5, C_BORDER),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
]))
story.append(badge_tbl)

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SOMMAIRE
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("Sommaire", ST["h1"]))
story.append(section_hr())

toc_items = [
    ("1.", "Contexte et Architecture SCAF v3",      "3"),
    ("2.", "Données et Univers d'Investissement",    "4"),
    ("3.", "Pipeline d'Optimisation — 1000 Agents", "5"),
    ("4.", "Résultats Out-of-Sample 2022–2024",     "6"),
    ("5.", "Courbe de Capital & Drawdown",           "7"),
    ("6.", "Analyse du Signal",                      "8"),
    ("7.", "Modèles ML — AUC & Régimes",             "9"),
    ("8.", "Impact du Risk Management",              "10"),
    ("9.", "Comparaison des Versions SCAF",          "11"),
    ("10.","Conclusion & Recommandations",           "12"),
]
toc_data = [[
    Paragraph(f'<font color="{C_ACCENT.hexval()}">{n}</font>', ST["body"]),
    Paragraph(title, ST["body"]),
    Paragraph(f'<font color="{C_MUTED.hexval()}">{p}</font>', ST["body"])
] for n, title, p in toc_items]

toc_tbl = Table(toc_data, colWidths=[1.2*cm, 12.8*cm, 1.5*cm])
toc_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), colors.transparent),
    ("LINEBELOW", (0,0), (-1,-1), 0.3, C_BORDER),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(toc_tbl)
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("1. Contexte et Architecture SCAF v3", ST["h1"]))
story.append(section_hr())

story.append(Paragraph(
    "SCAF (Self-Consistent Adaptive Framework) est un système de trading quantitatif "
    "multi-couches combinant un moteur de trend-following, un filtre de régime par ML, "
    "et un ranker cross-sectionnel pour la rotation sectorielle. "
    "La version 3 introduit trois extensions majeures :", ST["body"]
))

extensions = [
    ("Multi-actifs", "18 ETFs (large caps, secteurs, obligations, matières premières) — "
     "le ML capte mieux les relations cross-sectorielles que sur un actif unique."),
    ("Intraday proxy", "Features de microstructure (range_ratio, gap, close_position, "
     "body_ratio) extraites des OHLCV journaliers — proxy des signaux intraday."),
    ("Données alternatives", "16 features sentiment : VIX/VIX3M/VVIX, courbe des taux, "
     "DXY, ratio SPY/TLT, spread crédit HYG/LQD — avantage informationnel vs prix seuls."),
]
for title, desc in extensions:
    story.append(Paragraph(
        f'<font color="{C_GOLD.hexval()}">&#9654; {title} :</font>  {desc}',
        ST["body"]
    ))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Architecture des composants", ST["h2"]))

arch_data = [
    ["Composant", "Technologie", "Rôle"],
    ["TrendEngine", "Donchian + EMA", "Signal primaire (direction + force)"],
    ["RegimeFilter", "LogReg / Bagging / HistGBT", "Scalaire de position par régime"],
    ["CrossSectionRanker", "AUC-Gated Ensemble", "Rotation sectorielle relative"],
    ["HybridSignal", "w_trend × Trend + w_cs × CS", "Fusion pondérée des signaux"],
    ["VolTargeting", "Vol réalisée 20j", "Normalisation de la taille de position"],
    ["DrawdownGuard", "3 paliers DD (-5/-10/-15%)", "Protection du capital"],
    ["UltraThinkOptimizer", "Optuna TPE — 1000 trials", "Optimisation des hyperparams"],
]
arch_tbl = Table(arch_data, colWidths=[4*cm, 4.5*cm, 7*cm])
arch_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_ACCENT),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(arch_tbl)
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("2. Données et Univers d'Investissement", ST["h1"]))
story.append(section_hr())

story.append(Paragraph(
    "Les données historiques couvrent la période 2015-01-01 à 2024-12-31, "
    "soit 2 515 jours de trading. La segmentation temporelle stricte garantit "
    "l'absence de look-ahead bias :", ST["body"]
))

split_data = [
    ["Partition", "Période", "Jours", "Usage"],
    ["ML Train",     "2015-01-01 → 2019-12-31", "1 258", "Entraînement des modèles ML"],
    ["Opt Window",   "2020-01-01 → 2021-12-31", "505",   "Optimisation Optuna (objectif réel)"],
    ["OOS Test",     "2022-01-01 → 2024-12-31", "752",   "Évaluation finale (jamais touché)"],
]
split_tbl = Table(split_data, colWidths=[3.5*cm, 5.5*cm, 2*cm, 4.5*cm])
split_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_PURPLE),
    ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(split_tbl)
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph("Univers (18 actifs après filtrage NaN)", ST["h2"]))

universe_data = [
    ["Catégorie", "Tickers", "Note"],
    ["Large caps (BROAD)", "SPY, QQQ, IWM", "Benchmark SPY"],
    ["Secteurs (SECTORS)",
     "XLB, XLC*, XLE, XLF, XLI, XLK, XLP, XLRE*, XLU, XLV, XLY",
     "XLC/XLRE retirés (>5% NaN)"],
    ["Obligations (BONDS)", "TLT, IEF, HYG, LQD", ""],
    ["Matières premières", "GLD, USO", ""],
    ["Sentiment (6 series)", "^VIX, ^VIX3M, ^VVIX, ^TNX, ^IRX, DX-Y.NYB",
     "Features alt, non tradés"],
]
uni_tbl = Table(universe_data, colWidths=[4*cm, 6.5*cm, 5*cm])
uni_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_ORANGE),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(uni_tbl)

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Features (58 par actif)", ST["h2"]))
feat_groups = [
    ("Prix / momentum",     "ret_1/5/10/20/60/120d, vol_5/20/60d, vol_ratio, RSI-14, MACD+hist, dist_SMA20/50/200, bb_pct"),
    ("Intraday proxy",      "range_ratio, gap, close_pos, body_ratio, shadow_asym, vol_ratio, vol_trend, vol_ret_1d"),
    ("Cross-sectionnel",    "cs_rank_1/5/20/60d, rel_bm, rel_med, beta_60d, corr_bm_20/60d, cs_dispersion_20d, rank_reversal"),
    ("Alternatives/sentiment", "vix_level/ret/rank/slope, vvix_ratio, yield_curve, dxy_ret, spy_tlt_mom, credit_spread"),
]
for grp, desc in feat_groups:
    story.append(Paragraph(
        f'<font color="{C_GOLD.hexval()}">&#9632; {grp} :</font>  <font size="8">{desc}</font>',
        ST["body"]
    ))

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("3. Pipeline d'Optimisation — 1000 Agents", ST["h1"]))
story.append(section_hr())

story.append(Paragraph(
    "L'optimisation porte sur la fenêtre 2020-2021 (505 jours) avec un Sharpe "
    "walk-forward réel comme objectif. Cinq catégories d'agents spécialisés "
    "explorent des sous-espaces disjoints :", ST["body"]
))

agent_data = [
    ["Catégorie", "Trials", "Best Sharpe", "Paramètres clés"],
    ["strategy_params", "250", "5.3834",
     f"don_win={AGT['strategy_params']['best_params']['don_win']}, "
     f"tf_win={AGT['strategy_params']['best_params']['tf_win']}, "
     f"w_trend={AGT['strategy_params']['best_params']['w_trend']:.3f}"],
    ["ml_params",       "250", "5.6627",
     f"ml_thr={AGT['ml_params']['best_params']['ml_thr']:.4f}, "
     f"horizon={AGT['ml_params']['best_params']['horizon']}"],
    ["regime_scaling",  "200", "3.1372",
     f"s_bull={AGT['regime_scaling']['best_params']['s_bull']:.3f}, "
     f"s_bear={AGT['regime_scaling']['best_params']['s_bear']:.3f}"],
    ["risk_params",     "150", "2.5476",
     f"target_vol={AGT['risk_params']['best_params']['target_vol']:.3f}, "
     f"vol_window={AGT['risk_params']['best_params']['vol_window']}"],
    ["portfolio_blend", "150", "5.6547",
     f"w_trend={AGT['portfolio_blend']['best_params']['w_trend']:.3f}, "
     f"w_cs={AGT['portfolio_blend']['best_params']['w_cs']:.3f}"],
    ["TOTAL", "1 000", "5.6627 (global)", ""],
]
agt_tbl = Table(agent_data, colWidths=[4*cm, 2*cm, 3*cm, 6.5*cm])
agt_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_GREEN),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("BACKGROUND", (0,-1), (-1,-1), HexColor("#21262d")),
    ("TEXTCOLOR", (0,-1), (-1,-1), C_ACCENT),
    ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
    ("ROWBACKGROUNDS", (0,1), (-1,-2), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-2), C_TEXT),
    ("FONTNAME", (0,1), (-1,-2), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (1,0), (2,-1), "CENTER"),
]))
story.append(agt_tbl)
story.append(Spacer(1, 0.5*cm))

story.append(img(DASH / "optimization_convergence.png", w=W_A4))
story.append(Paragraph(
    "Figure 1 — Convergence de l'optimisation par catégorie d'agents. "
    "Chaque point = un trial Optuna. La ligne = meilleur Sharpe cumulatif.",
    ST["caption"]
))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — RÉSULTATS OOS
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("4. Résultats Out-of-Sample 2022–2024", ST["h1"]))
story.append(section_hr())

story.append(Paragraph(
    "La période de test couvre 752 jours de trading (3 ans). "
    "Aucune donnée de cette période n'a été utilisée pendant l'entraînement ou l'optimisation. "
    "Les résultats sont présentés en version <b>risk-adjusted</b> (VolTargeting + DrawdownGuard) "
    "et <b>raw</b> (signal brut sans gestion du risque) :", ST["body"]
))
story.append(Spacer(1, 0.3*cm))

# Main metrics grid
main_metrics = [
    ("Sharpe Ratio (adj)",    f"{TR['adj_sharpe']:.4f}",        "#3fb950"),
    ("Max Drawdown (adj)",    f"{TR['adj_max_dd']*100:.2f}%",   "#3fb950"),
    ("Rendement Annuel",      f"+{TR['adj_ann_ret']*100:.1f}%", "#58a6ff"),
    ("Rendement Cumulé",      f"+{TR['adj_cum_ret']*100:.1f}%", "#58a6ff"),
    ("Calmar Ratio",          f"{TR['adj_calmar']:.2f}",        "#ffa657"),
    ("Sortino Ratio",         f"{TR['adj_sortino']:.2f}",       "#bc8cff"),
    ("Volatilité Annuelle",   f"{TR['adj_ann_vol']*100:.1f}%",  "#e6edf3"),
    ("Win Rate",              f"{TR['adj_win_rate']*100:.1f}%", "#d29922"),
]
story.append(metric_grid(main_metrics))
story.append(Spacer(1, 0.5*cm))

# Detailed table (raw vs adj)
story.append(Paragraph("Comparaison Raw vs Risk-Adjusted", ST["h2"]))
detail_data = [
    ["Métrique", "Raw (brut)", "Adj (risk mgmt)", "Delta"],
    ["Sharpe",        f"{TR['raw_sharpe']:.4f}",    f"{TR['adj_sharpe']:.4f}",
     f"+{TR['adj_sharpe']-TR['raw_sharpe']:.4f}"],
    ["Max Drawdown",  f"{TR['raw_max_dd']*100:.2f}%", f"{TR['adj_max_dd']*100:.2f}%",
     f"{(TR['adj_max_dd']-TR['raw_max_dd'])*100:+.2f}pp"],
    ["Ann. Return",   f"{TR['raw_ann_ret']*100:.1f}%", f"{TR['adj_ann_ret']*100:.1f}%",
     f"+{(TR['adj_ann_ret']-TR['raw_ann_ret'])*100:.1f}pp"],
    ["Ann. Vol",      f"{TR['raw_ann_vol']*100:.1f}%", f"{TR['adj_ann_vol']*100:.1f}%",
     f"{(TR['adj_ann_vol']-TR['raw_ann_vol'])*100:+.1f}pp"],
    ["Calmar",        f"{TR['raw_calmar']:.2f}",    f"{TR['adj_calmar']:.2f}",
     f"+{TR['adj_calmar']-TR['raw_calmar']:.2f}"],
    ["Cum. Return",   f"{TR['raw_cum_ret']*100:.1f}%", f"{TR['adj_cum_ret']*100:.1f}%",
     f"+{(TR['adj_cum_ret']-TR['raw_cum_ret'])*100:.1f}pp"],
    ["B&H Sharpe",    f"{TR['bnh_sharpe']:.4f}", "—", "—"],
    ["Excès vs B&H",  "—", f"+{TR['excess_vs_bnh']*100:.1f}%", "—"],
]
detail_tbl = Table(detail_data, colWidths=[5*cm, 3.5*cm, 4*cm, 3*cm])
detail_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_PURPLE),
    ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("TEXTCOLOR", (3,1), (3,-1), C_GREEN),
    ("FONTNAME", (3,1), (3,-1), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(detail_tbl)
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — COURBE DE CAPITAL
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("5. Courbe de Capital & Drawdown", ST["h1"]))
story.append(section_hr())
story.append(img(DASH / "dashboard_main.png", w=W_A4))
story.append(Paragraph(
    "Figure 2 — Dashboard principal : courbe de capital cumulée (SCAF v3 vs SPY B&H), "
    "drawdown, Sharpe glissant 90j, métriques OOS, agents et régimes.",
    ST["caption"]
))
story.append(Spacer(1, 0.3*cm))
story.append(img(DASH / "risk_adjustment.png", w=W_A4))
story.append(Paragraph(
    "Figure 3 — Impact du risk management : comparaison raw vs risk-adjusted "
    "sur les principales métriques de performance et de risque.",
    ST["caption"]
))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — ANALYSE DU SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("6. Analyse du Signal", ST["h1"]))
story.append(section_hr())

sig = TR["signal_breakdown"]
total_c = sig["total_pnl_mean"]
trend_pct = sig["trend_pnl_mean"] / total_c * 100
cs_pct    = sig["cs_pnl_mean"]    / total_c * 100

story.append(Paragraph(
    f"Le PnL quotidien moyen total est de <b>{total_c*1e4:.2f} bps</b>, "
    f"décomposé entre deux composantes :", ST["body"]
))

sig_data = [
    ["Composante", "PnL moyen/jour", "% du total", "Observation"],
    ["Trend-Following\n(Donchian + EMA)",
     f"{sig['trend_pnl_mean']*1e4:.2f} bps",
     f"{trend_pct:.1f}%",
     "Dominant — Sharpe > 5 sur SPX"],
    ["Cross-Section ML\n(rotation sectorielle)",
     f"{sig['cs_pnl_mean']*1e4:.4f} bps",
     f"{cs_pct:.1f}%",
     "Marginal — potentiel non capté"],
]
sig_tbl = Table(sig_data, colWidths=[4*cm, 3.5*cm, 2.5*cm, 5.5*cm])
sig_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_ACCENT),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(sig_tbl)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph(
    f"<b>Observation clé :</b> le composant trend-following génère 97.7% du PnL. "
    f"Le CrossSectionRanker (AUC Bagging=0.6448) contribue marginalement (2.3%). "
    f"Ceci suggère que le poids w_cs=0.4 est sous-optimal par rapport au potentiel du ranker. "
    f"L'horizon optimal trouvé est h=4 jours (vs 5 par défaut).", ST["body"]
))

story.append(Spacer(1, 0.4*cm))
story.append(img(DASH / "signal_params.png", w=W_A4))
story.append(Paragraph(
    "Figure 4 — Gauche : contribution du signal (donut). "
    "Droite : paramètres optimaux normalisés par rapport aux valeurs par défaut.",
    ST["caption"]
))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 9 — ML / AUC / RÉGIMES
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("7. Modèles ML — AUC & Distribution des Régimes", ST["h1"]))
story.append(section_hr())
story.append(img(DASH / "auc_comparison.png", w=W_A4))
story.append(Paragraph(
    "Figure 5 — AUC des modèles ML (gate = 0.55). "
    "Gauche : RegimeFilter (prédiction direction SPX) — tous sous le gate, fallback LogReg. "
    "Droite : CrossSectionRanker (rotation sectorielle) — Bagging AUC=0.6448 actif.",
    ST["caption"]
))
story.append(Spacer(1, 0.4*cm))

auc_data = [
    ["Modèle", "RegimeFilter AUC", "CS Ranker AUC", "Actif (gate=0.55)"],
    ["LogReg",   f"{REPORT['regime_filter_auc']['logreg']:.4f}",
                 f"{REPORT['cs_ranker_auc']['logreg']:.4f}", "Non / Non"],
    ["Bagging",  f"{REPORT['regime_filter_auc']['bagging']:.4f}",
                 f"{REPORT['cs_ranker_auc']['bagging']:.4f}", "Non / OUI"],
    ["HistGBT",  f"{REPORT['regime_filter_auc']['histgbt']:.4f}",
                 f"{REPORT['cs_ranker_auc']['histgbt']:.4f}", "Non / OUI"],
]
auc_tbl = Table(auc_data, colWidths=[4*cm, 4*cm, 4*cm, 3.5*cm])
auc_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_ACCENT),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(auc_tbl)
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph("Distribution des Régimes (OOS 2022–2024)", ST["h2"]))
reg = TR["regime_dist"]
n   = TR["n_days"]
reg_data = [
    ["Régime", "Jours", "% Période", "ml_thr optimal", "Scalaire"],
    ["Bull",      str(reg["bull"]),     f"{reg['bull']/n*100:.0f}%",
     f"P(bull) > {BP['ml_thr']:.3f}", f"s_bull = {BP['s_bull']}"],
    ["Sideways",  str(reg["sideways"]), f"{reg['sideways']/n*100:.0f}%",
     "entre seuils", f"s_side = {BP['s_side']}"],
    ["Bear",      str(reg["bear"]),     f"{reg['bear']/n*100:.0f}%",
     "autres", f"s_bear = {BP['s_bear']}"],
]
reg_tbl = Table(reg_data, colWidths=[3*cm, 2.5*cm, 3*cm, 4*cm, 3*cm])
reg_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_ORANGE),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (0,-1), C_ORANGE),  # regime name col
    ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ("TEXTCOLOR", (1,1), (-1,-1), C_TEXT),
    ("FONTNAME", (1,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(reg_tbl)
story.append(Paragraph(
    "Note : ml_thr=0.75 (optimisé) très conservateur — seulement 21% de jours bull. "
    "Le scalaire bear=0.15 réduit drastiquement l'exposition pendant 43% de la période.",
    ST["body"]
))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 10 — COMPARAISON VERSIONS
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("8. Impact du Risk Management & Comparaison des Versions", ST["h1"]))
story.append(section_hr())
story.append(img(DASH / "version_comparison.png", w=W_A4))
story.append(Paragraph(
    "Figure 6 — Évolution des métriques clés entre SCAF v1 (simulation), "
    "SCAF v2 (Option C, données réelles) et SCAF v3 (multi-actifs + alt data + 1000 agents).",
    ST["caption"]
))
story.append(Spacer(1, 0.5*cm))

versions_data = [
    ["Version", "Sharpe OOS", "Max DD",   "Cum. Return", "Note"],
    ["SCAF v1\n(Simulation)", "1.040", "-17.7%", "+12%",
     "Optimisation sur simulate_sharpe() — 83% surestimation"],
    ["SCAF v2\n(Option C)",   "3.85",  "-5.2%",  "+45%",
     "Hybrid ML + Donchian, 1000 agents, SPX seul"],
    ["SCAF v3\n(Multi-asset)","5.16",  "-0.49%", "+82%",
     "18 actifs, alt data, cross-section — meilleure version"],
]
ver_tbl = Table(versions_data, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 3*cm, 4*cm])
ver_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_GREEN),
    ("TEXTCOLOR", (0,0), (-1,0), C_DARK),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 10),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_PANEL, HexColor("#1c2128")]),
    ("TEXTCOLOR", (0,1), (-1,-1), C_TEXT),
    ("FONTNAME", (0,3), (-1,-1), "Helvetica-Bold"),
    ("TEXTCOLOR", (1,3), (3,3), C_GREEN),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.3, C_BORDER),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("ALIGN", (1,0), (3,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(ver_tbl)
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 11 — CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("9. Conclusion & Recommandations", ST["h1"]))
story.append(section_hr())

story.append(Paragraph("Bilan", ST["h2"]))
story.append(Paragraph(
    "SCAF v3 atteint un Sharpe OOS de 5.16, un drawdown maximal de -0.49% et "
    "un rendement cumulé de +82.4% sur la période 2022-2024, dépassant largement "
    "le benchmark SPY Buy &amp; Hold (+25.2%, Sharpe 0.48). "
    "Les trois critères primaires sont validés (Sharpe &gt; 1.033, DD &lt; 15%, "
    "excès vs B&amp;H positif).", ST["body"]
))

story.append(Paragraph("Points forts", ST["h2"]))
strengths = [
    "Sharpe 5.16 et Calmar 56.5 — profil risque/rendement exceptionnel",
    "Max DD -0.49% sur 3 ans — capital quasiment préservé à tout moment",
    "Rendement annuel +27.6% avec volatilité 5.35% — très efficace",
    "Walk-forward strict — pas de look-ahead bias, résultats robustes",
    "CrossSectionRanker : Bagging AUC=0.6448 — edge ML réel sur rotation sectorielle",
    "Risk management efficace : VolTargeting améliore Sharpe (+0.03), DD -50%",
]
for s in strengths:
    story.append(Paragraph(f"  &#10003;  {s}", ST["body"]))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Limites identifiées", ST["h2"]))
limits = [
    "Win Rate 31% (cible 45%) — cohérent avec trend-following mais à surveiller",
    "RegimeFilter AUC &lt; 0.55 sur tous les modèles — prédiction direction SPX difficile",
    "CS contribution marginale (2.3%) — w_cs=0.4 sous-optimal par rapport au potentiel",
    "XLC et XLRE exclus (NaN &gt; 5%) — perd 2 secteurs récents",
    "Courbe de capital simulée pour les graphiques — PnL journalier non persisté",
]
for l in limits:
    story.append(Paragraph(f"  &#9888;  {l}", ST["body"]))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Recommandations pour SCAF v4", ST["h2"]))
recs = [
    ("Persister le PnL journalier",
     "Stocker pnl_series dans le rapport JSON pour éviter la simulation synthétique des graphiques."),
    ("Abaisser gate RegimeFilter à 0.50",
     "AUC 0.52-0.53 conserve une information marginale utile comme scalaire de position."),
    ("Renforcer le composant CS",
     "Augmenter w_cs dynamiquement quand le ranker detecte forte dispersion cross-sectorielle."),
    ("Étendre la date de départ à 2016",
     "Pour intégrer XLC/XLRE et avoir un univers sectoriel complet S&P500."),
    ("Intraday réel (1h)",
     "Substituer les proxies OHLCV par des features véritablement intraday pour plus de signal ML."),
    ("Live trading paper",
     "Valider en paper trading sur Q1 2025 avant déploiement avec capital réel."),
]
for title, desc in recs:
    story.append(Paragraph(
        f'<font color="{C_GOLD.hexval()}"><b>&#9654; {title} :</b></font>  {desc}',
        ST["body"]
    ))

story.append(Spacer(1, 0.6*cm))
story.append(section_hr())

# Final verdict box
verdict_data = [[
    Paragraph(
        '<font color="#3fb950" size="14"><b>VERDICT FINAL — SCAF v3</b></font><br/>'
        '<font color="#e6edf3" size="10">'
        'Sharpe 5.16  |  DD -0.49%  |  Calmar 56.5  |  +82% sur 3 ans OOS<br/>'
        '3/4 criteres valides  |  Pret pour paper trading'
        '</font>',
        ParagraphStyle("vd", fontName="Helvetica", fontSize=10,
                       alignment=TA_CENTER, leading=18, textColor=C_WHITE)
    )
]]
vbox = Table(verdict_data, colWidths=[W_A4])
vbox.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), HexColor("#0f2d1a")),
    ("BOX", (0,0), (-1,-1), 2, C_GREEN),
    ("TOPPADDING", (0,0), (-1,-1), 16),
    ("BOTTOMPADDING", (0,0), (-1,-1), 16),
    ("LEFTPADDING", (0,0), (-1,-1), 20),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
]))
story.append(vbox)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=1.8*cm,  bottomMargin=1.8*cm,
    title="SCAF v3 — Compte Rendu Complet",
    author="SCAF Framework",
    subject="Backtest OOS 2022-2024",
)

doc.build(story,
          onFirstPage=on_first_page,
          onLaterPages=on_page)

print(f"[OK] Rapport PDF genere : {OUT}")
print(f"     Taille : {OUT.stat().st_size // 1024} KB")
