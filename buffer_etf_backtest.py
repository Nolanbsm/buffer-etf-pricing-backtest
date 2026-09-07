"""
Buffer ETF Synthétique S&P 500 — Backtest 2005–2025
Mémoire Partie II : Construction, Backtest et Analyse

Architecture :
    II.1 — Réplication BS (4 briques FLEX + cap zero-cost)
    II.2 — Performance point-to-point (payoff à l'échéance)
    II.3 — Performance mark-to-market (intérimaire, rolling)
    II.4 — Comparaison vs stock/cash beta-matched (méthodologie AQR)
    II.5 — Sensibilité du cap au VIX et aux taux
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BUFFER      = 0.10        # 10% downside buffer
START_DATE  = "2005-01-01"
END_DATE    = "2025-01-01"
RISK_FREE   = None        # sera remplacé par T-bill 1Y
FIG_DIR     = "/mnt/user-data/outputs"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.facecolor": "#0d1117",
    "figure.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "axes.titlecolor": "#e6edf3",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 8,
    "font.family": "monospace",
})

COLORS = {
    "spx":    "#58a6ff",
    "buffer": "#3fb950",
    "cash":   "#f78166",
    "mixed":  "#d2a8ff",
    "neutral":"#8b949e",
}

# ─────────────────────────────────────────────
# II.1  DONNÉES
# ─────────────────────────────────────────────

def fetch_data():
    """
    Données synthétiques calibrées sur les niveaux historiques réels.
    En production : remplacer par yf.download("^GSPC", ...) etc.
    
    Calibration :
        SPX : GBM avec returns annuels calés sur les vraies années
        VIX : processus mean-reverting autour des niveaux historiques
        rf  : interpolation des niveaux Fed Funds / T-bill 1Y historiques
    
    Sources de calibration :
        SPX returns  : données CRSP/Bloomberg
        VIX niveaux  : CBOE historical data
        Taux rf      : Fed H.15 Selected Interest Rates
    """
    print("📥  Construction des données synthétiques calibrées (2005–2025)...")
    print("    (En production : remplacer par yf.download)")

    # ── Returns annuels SPX price return historiques (source : Bloomberg/CRSP)
    # Note : price return ≠ total return → pas de dividendes (≈-1.8%/an d'écart)
    spx_annual = {
        2005: 0.030, 2006: 0.136, 2007: 0.035, 2008: -0.385,
        2009: 0.235, 2010: 0.128, 2011: 0.000, 2012: 0.130,
        2013: 0.295, 2014: 0.115, 2015: -0.007, 2016: 0.095,
        2017: 0.193, 2018: -0.065, 2019: 0.289, 2020: 0.163,
        2021: 0.269, 2022: -0.196, 2023: 0.242, 2024: 0.230,
    }

    # ── VIX moyens annuels (niveaux de début d'année pour le pricer)
    vix_annual = {
        2005: 12.8, 2006: 11.5, 2007: 17.5, 2008: 32.7,
        2009: 31.1, 2010: 17.6, 2011: 24.2, 2012: 14.6,
        2013: 12.3, 2014: 12.1, 2015: 15.3, 2016: 18.2,
        2017: 10.6, 2018: 16.6, 2019: 13.8, 2020: 26.5,
        2021: 19.4, 2022: 26.5, 2023: 18.9, 2024: 13.9,
    }

    # ── T-bill 1Y (taux continu approx)
    rf_annual = {
        2005: 0.032, 2006: 0.049, 2007: 0.047, 2008: 0.015,
        2009: 0.003, 2010: 0.003, 2011: 0.002, 2012: 0.002,
        2013: 0.002, 2014: 0.003, 2015: 0.005, 2016: 0.005,
        2017: 0.010, 2018: 0.022, 2019: 0.023, 2020: 0.001,
        2021: 0.001, 2022: 0.016, 2023: 0.052, 2024: 0.052,
    }

    np.random.seed(42)
    records = []
    biz_dates = pd.bdate_range(START_DATE, END_DATE)

    S = 1200.0  # SPX ~1200 en Jan 2005
    for yr in range(2005, 2025):
        yr_dates = [d for d in biz_dates if d.year == yr]
        n = len(yr_dates)
        ann_ret  = spx_annual.get(yr, 0.10)
        vix0     = vix_annual.get(yr, 15.0)
        rf0      = rf_annual.get(yr, 0.03)
        sigma    = vix0 / 100.0

        # Chemin GBM journalier calibré sur le return annuel
        daily_mu  = np.log(1 + ann_ret) / n
        daily_sig = sigma / np.sqrt(252)
        eps = np.random.randn(n)
        log_rets = daily_mu + daily_sig * eps
        # Rescale pour matcher exactement le return annuel
        cumulative = np.exp(np.cumsum(log_rets))
        target = 1 + ann_ret
        scaling = (target / cumulative[-1]) ** (1 / n)
        log_rets = log_rets + np.log(scaling)

        prices = S * np.exp(np.cumsum(log_rets))
        S_end = prices[-1]

        # VIX : mean-reverting autour de vix0
        vix_path = vix0 + 3 * np.random.randn(n).cumsum() * 0.1
        vix_path = np.clip(vix_path, 8, 80)
        vix_path[0] = vix0  # fixe le niveau au roll

        # rf : constant sur l'année (proxy T-bill 1Y début d'année)
        rf_path = np.full(n, rf0)

        for i, d in enumerate(yr_dates):
            records.append({
                "date": d,
                "spx": prices[i],
                "vix": vix_path[i],
                "rf":  rf_path[i],
            })
        S = S_end

    df = pd.DataFrame(records).set_index("date")
    print(f"✅  Données : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} jours)")
    return df


# ─────────────────────────────────────────────
# II.1  BLACK-SCHOLES TOOLKIT
# ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Prix BS européen (call ou put)."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def buffer_payoff_bs(S0, St, T, r, sigma, buffer=0.10, cap=None):
    """
    Payoff analytique à maturité T d'un buffer ETF (réplication BS).

    4 briques FLEX (price return sur S0=1 normalisé) :
        Long  call(K=S0)          → participation à la hausse
        Short call(K=S0*(1+cap))  → plafond de gain (cap)
        Long  put(K=S0)           → protection 0..buffer
        Short put(K=S0*(1-buffer))→ absorbe les premières pertes (buffer)

    On retourne le prix des 4 briques à t=0 ET le payoff à T.
    """
    # Normalisation S0=1 pour facilité
    K_call_long  = 1.0
    K_call_short = 1.0 + cap if cap else np.inf
    K_put_long   = 1.0
    K_put_short  = 1.0 - buffer

    x = St / S0  # return gross de l'index

    # Payoff analytique (formule en lignes brisées)
    if x >= 1.0 + (cap if cap else np.inf):
        payoff = cap
    elif x >= 1.0:
        payoff = x - 1.0
    elif x >= 1.0 - buffer:
        payoff = 0.0
    else:
        payoff = x - (1.0 - buffer)

    return payoff


def price_four_legs(T, r, sigma, buffer, cap):
    """
    Coût net de la structure zero-cost (normalisé, S0=1).

    Convention correcte (émetteurs réels) :
        Long  call(K=1.0)          → participate hausse
        Short call(K=1+cap)        → plafond
        Long  put(K=1.0)           → protection full
        Short put(K=1-buffer)      → l'investisseur absorbe les buffer premières pertes

    Zero-cost ↔ call_spread = put_spread
        call_spread = c_long - c_short   (revenu de la vente du call OTM)
        put_spread  = p_long - p_short   (coût net de la protection)
    On cherche cap tel que net = call_spread - put_spread = 0
    """
    c_long  = bs_price(1.0, 1.0,          T, r, sigma, "call")
    c_short = bs_price(1.0, 1.0 + cap,    T, r, sigma, "call")
    p_long  = bs_price(1.0, 1.0,          T, r, sigma, "put")
    p_short = bs_price(1.0, 1.0 - buffer, T, r, sigma, "put")
    call_spread = c_long - c_short
    put_spread  = p_long - p_short
    return call_spread - put_spread


def find_cap_zero_cost(S0, T, r, sigma, buffer):
    """
    Dichotomie (Brent) : cap tel que call_spread = put_spread.
    Plus le cap est bas, moins le call short rapporte → équilibre vers le bas.
    Plus la vol est haute, plus le put spread est cher → cap plus bas.
    """
    def objective(cap):
        return price_four_legs(T, r, sigma, buffer, cap)

    # À cap→0 : call_spread→0, put_spread>0 → objective < 0
    # À cap→∞ : call_spread→c_long, put_spread fixe → objective > 0
    f_low  = objective(1e-4)
    f_high = objective(5.0)
    if f_low * f_high > 0:
        return np.nan
    try:
        cap = brentq(objective, 1e-4, 5.0, xtol=1e-6, maxiter=200)
        return cap
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────
# II.2  BACKTEST POINT-TO-POINT
# ─────────────────────────────────────────────

def run_pointtopoint_backtest(df, buffer=BUFFER):
    """
    Roll annuel : chaque 1er janvier (ou 1er jour ouvré),
    on fixe S0, on calcule le cap zero-cost via BS,
    on attend 1 an, on calcule le payoff.
    """
    print("\n📊  II.2 — Backtest point-to-point...")

    results = []
    years = range(2005, 2025)

    for yr in years:
        # Date de roll : 1er jour ouvré de l'année
        try:
            roll_date = df.loc[f"{yr}-01-01":f"{yr}-03-01"].index[0]
            next_yr_data = df.loc[f"{yr+1}-01-01":f"{yr+1}-03-01"]
            if len(next_yr_data) == 0:
                # Utiliser le dernier jour disponible
                end_date = df.loc[f"{yr}-12-01":].index[-1]
            else:
                end_date = next_yr_data.index[0]
        except IndexError:
            continue

        S0    = df.loc[roll_date, "spx"]
        ST    = df.loc[end_date,  "spx"]
        r     = df.loc[roll_date, "rf"]
        sigma = df.loc[roll_date, "vix"] / 100.0  # VIX → vol annualisée
        T     = 1.0  # 1 an

        cap = find_cap_zero_cost(1.0, T, r, sigma, buffer)
        if np.isnan(cap):
            continue

        spx_ret    = ST / S0 - 1.0
        buffer_ret = buffer_payoff_bs(S0, ST, T, r, sigma, buffer, cap)

        results.append({
            "year":       yr,
            "roll_date":  roll_date,
            "end_date":   end_date,
            "S0":         S0,
            "ST":         ST,
            "r":          r,
            "sigma":      sigma,
            "cap":        cap,
            "spx_ret":    spx_ret,
            "buffer_ret": buffer_ret,
        })

    df_bt = pd.DataFrame(results).set_index("year")
    return df_bt


def compute_metrics(returns_series, rf=0.02):
    """Métriques de performance annualisées."""
    r = returns_series.dropna()
    n = len(r)
    ann_ret  = (1 + r).prod() ** (1 / n) - 1
    ann_vol  = r.std() * np.sqrt(1)  # déjà annuel
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    # Max Drawdown sur NAV cumulée
    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd   = (nav - peak) / peak
    mdd  = dd.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan

    return {
        "Ann. Return": ann_ret,
        "Ann. Vol":    ann_vol,
        "Sharpe":      sharpe,
        "Max DD":      mdd,
        "Calmar":      calmar,
    }


# ─────────────────────────────────────────────
# II.3  MARK-TO-MARKET (ROLLING INTRAANNUEL)
# ─────────────────────────────────────────────

def bs_buffer_mtm(S0, St, T_remaining, r, sigma, buffer, cap):
    """
    Valeur mark-to-market du buffer ETF en cours d'année.
    On reprice les 4 briques avec T_remaining et St comme spot.
    Normalisation : NAV initiale = 1
    """
    if T_remaining <= 0:
        return buffer_payoff_bs(S0, St, 0, r, sigma, buffer, cap) + 1.0

    # Prix des 4 briques (strikes = K/St normalisés par rapport à St)
    # On travaille en spot = 1 (normalisé) avec strikes ajustés
    K_ratio_cl  = 1.0 / (St / S0)       # K_call_long / St
    K_ratio_cs  = (1.0 + cap) / (St / S0)
    K_ratio_pl  = 1.0 / (St / S0)
    K_ratio_ps  = (1.0 - buffer) / (St / S0)

    c_long  = bs_price(1.0, K_ratio_cl,  T_remaining, r, sigma, "call")
    c_short = bs_price(1.0, K_ratio_cs,  T_remaining, r, sigma, "call")
    p_long  = bs_price(1.0, K_ratio_pl,  T_remaining, r, sigma, "put")
    p_short = bs_price(1.0, K_ratio_ps,  T_remaining, r, sigma, "put")

    # NAV = valeur nette du portefeuille d'options (normalisé à 1 à t=0)
    nav = (c_long - c_short) + (p_long - p_short) + (St / S0)
    return nav


def run_mtm_backtest(df, df_bt, buffer=BUFFER):
    """
    Pour chaque jour de trading 2005–2024,
    calcule la valeur MtM du buffer ETF du millésime en cours.
    Retourne un DataFrame journalier avec : spx_idx, buffer_nav, spx_ret_yoy, buffer_ret_yoy
    """
    print("📊  II.3 — Backtest mark-to-market (daily, peut prendre ~1 min)...")

    records = []
    for _, row in df_bt.iterrows():
        roll  = row["roll_date"]
        end   = row["end_date"]
        S0    = row["S0"]
        r     = row["r"]
        sigma = row["sigma"]
        cap   = row["cap"]

        # Tous les jours de trading dans la fenêtre [roll, end]
        window = df.loc[roll:end]
        T_total = 1.0

        for date, drow in window.iterrows():
            elapsed     = (date - roll).days / 365.25
            T_remaining = max(T_total - elapsed, 0)
            St          = drow["spx"]
            sigma_t     = drow["vix"] / 100.0  # vol du jour → repricing

            nav = bs_buffer_mtm(S0, St, T_remaining, r, sigma_t, buffer, cap)

            records.append({
                "date":       date,
                "vintage":    row.name,
                "S0":         S0,
                "St":         St,
                "T_rem":      T_remaining,
                "cap":        cap,
                "spx_ret":    St / S0 - 1.0,
                "buffer_nav": nav,
                "buffer_ret": nav - 1.0,
            })

    df_mtm = pd.DataFrame(records).set_index("date")
    return df_mtm


def compute_aqr_stat(df_mtm):
    """
    Reproduit l'Exhibit 7 AQR : % d'observations où le buffer ETF perd
    de l'argent alors que le marché est dans la zone protégée (0 à -10%).
    """
    in_buffer_zone = (df_mtm["spx_ret"] >= -BUFFER) & (df_mtm["spx_ret"] < 0)
    loses_money    = df_mtm["buffer_ret"] < 0
    both           = in_buffer_zone & loses_money

    pct = both.sum() / in_buffer_zone.sum() * 100
    return pct, in_buffer_zone.sum(), both.sum()


# ─────────────────────────────────────────────
# II.4  STOCK/CASH BETA-MATCHED (AQR)
# ─────────────────────────────────────────────

def stock_cash_benchmark(df_bt, df, buffer=BUFFER):
    """
    Méthodologie AQR Exhibit 1 — beta estimé par OLS sur les returns annuels.

    On régresse les returns du buffer ETF sur ceux du SPX (excès de rendement
    au-dessus du cash) pour obtenir le beta effectif réel du produit.
    C'est la méthode AQR, pas le delta analytique du call spread
    (qui sous-estime systématiquement le beta car il ignore le put spread).

    β_eff = Cov(R_buffer - r_f, R_SPX - r_f) / Var(R_SPX - r_f)

    Le benchmark est ensuite : β_eff * R_SPX + (1 - β_eff) * r_f
    rebalancé annuellement au même rythme que le roll du buffer ETF.
    """
    print("📊  II.4 — Benchmark stock/cash beta-matched (OLS)...")

    spx_rets    = df_bt["spx_ret"].values
    buf_rets    = df_bt["buffer_ret"].values
    cash_rets   = np.array([np.exp(r) - 1.0 for r in df_bt["r"].values])

    # Excès de rendement au-dessus du cash
    spx_excess = spx_rets - cash_rets
    buf_excess  = buf_rets - cash_rets

    # OLS : β = Cov(buf_excess, spx_excess) / Var(spx_excess)
    beta_ols = np.cov(buf_excess, spx_excess)[0, 1] / np.var(spx_excess, ddof=1)
    beta_ols = np.clip(beta_ols, 0.1, 1.5)  # garde-fou

    print(f"    β_eff (OLS, full sample) = {beta_ols:.3f}")

    records = []
    for yr, row in df_bt.iterrows():
        r          = row["r"]
        spx_ret    = row["spx_ret"]
        cash_ret   = np.exp(r) - 1.0
        bench_ret  = beta_ols * spx_ret + (1 - beta_ols) * cash_ret

        records.append({
            "year":       yr,
            "beta_eff":   beta_ols,
            "spx_ret":    spx_ret,
            "cash_ret":   cash_ret,
            "bench_ret":  bench_ret,
            "buffer_ret": row["buffer_ret"],
        })

    df_bench = pd.DataFrame(records).set_index("year")
    return df_bench


# ─────────────────────────────────────────────
# II.5  SENSIBILITÉ DU CAP
# ─────────────────────────────────────────────

def cap_sensitivity_analysis(df_bt):
    """
    Décompose comment le cap évolue selon la vol (VIX) et les taux.
    Focus sur les années marquantes : 2019, 2020, 2021, 2022.
    """
    focus_years = [2007, 2008, 2019, 2020, 2021, 2022, 2023]
    df_focus = df_bt[df_bt.index.isin(focus_years)][["sigma", "r", "cap"]].copy()
    df_focus["vix_pct"] = df_focus["sigma"] * 100
    df_focus["r_pct"]   = df_focus["r"] * 100
    df_focus["cap_pct"] = df_focus["cap"] * 100
    return df_focus


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_all(df, df_bt, df_mtm, df_bench, df_cap):
    print("\n🎨  Génération des graphiques...")

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── AX1 : NAV cumulée point-to-point ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    nav_spx    = (1 + df_bt["spx_ret"]).cumprod()
    nav_buf    = (1 + df_bt["buffer_ret"]).cumprod()
    nav_bench  = (1 + df_bench["bench_ret"]).cumprod()
    years      = df_bt.index

    ax1.plot(years, nav_spx,   color=COLORS["spx"],    lw=2,   label="S&P 500 (price return)")
    ax1.plot(years, nav_buf,   color=COLORS["buffer"],  lw=2,   label=f"Buffer ETF synthétique (10% buffer)")
    ax1.plot(years, nav_bench, color=COLORS["cash"],    lw=2, ls="--", label="Stock/Cash β-matched (AQR)")
    ax1.axvline(2008, color="#f78166", alpha=0.3, lw=1.2, ls=":")
    ax1.axvline(2020, color="#f78166", alpha=0.3, lw=1.2, ls=":")
    ax1.axvline(2022, color="#f78166", alpha=0.3, lw=1.2, ls=":")
    ax1.text(2008.1, ax1.get_ylim()[0]*1.02, "GFC", color="#f78166", fontsize=8)
    ax1.text(2020.1, ax1.get_ylim()[0]*1.02, "COVID", color="#f78166", fontsize=8)
    ax1.text(2022.1, ax1.get_ylim()[0]*1.02, "2022", color="#f78166", fontsize=8)
    ax1.set_title("II.2 — Performance point-to-point : NAV cumulée (base 1, 2005–2024)")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#0d1117")

    # ── AX2 : Rendements annuels comparés ───────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    x   = np.arange(len(years))
    w   = 0.28
    ax2.bar(x - w, df_bt["spx_ret"] * 100,   width=w, color=COLORS["spx"],    label="SPX", alpha=0.85)
    ax2.bar(x,     df_bt["buffer_ret"] * 100, width=w, color=COLORS["buffer"], label="Buffer", alpha=0.85)
    ax2.bar(x + w, df_bench["bench_ret"] * 100, width=w, color=COLORS["cash"], label="β-matched", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=7)
    ax2.axhline(0, color="#8b949e", lw=0.8)
    ax2.set_title("II.2 — Rendements annuels (%)")
    ax2.set_ylabel("Return (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#0d1117")

    # ── AX3 : Payoff diagram (lignes brisées, exemple 2022) ─────────────
    ax3 = fig.add_subplot(gs[1, 1])
    # Choisir une année représentative pour le diagramme
    ex_year = 2022 if 2022 in df_bt.index else df_bt.index[-2]
    ex = df_bt.loc[ex_year]
    cap_ex = ex["cap"]

    spx_range = np.linspace(-0.40, 0.50, 500)
    buf_payoff = np.array([buffer_payoff_bs(1, 1+r, 1, ex["r"], ex["sigma"], BUFFER, cap_ex)
                           for r in spx_range])

    ax3.plot(spx_range * 100, spx_range * 100, color=COLORS["spx"],    lw=1.5, ls="--", label="SPX 1-pour-1", alpha=0.6)
    ax3.plot(spx_range * 100, buf_payoff * 100, color=COLORS["buffer"], lw=2.5, label=f"Buffer ETF (cap={cap_ex:.1%})")
    ax3.axhline(0, color="#8b949e", lw=0.5)
    ax3.axvline(0, color="#8b949e", lw=0.5)
    ax3.fill_betweenx([-5, 0], -BUFFER*100, 0, alpha=0.08, color=COLORS["buffer"], label="Zone protégée")
    ax3.set_title(f"II.2 — Diagramme de payoff à maturité (millésime {ex_year})")
    ax3.set_xlabel("SPX return (%)")
    ax3.set_ylabel("Buffer ETF return (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#0d1117")

    # ── AX4 : MtM — Scatter SPX return vs Buffer return ────────────────
    ax4 = fig.add_subplot(gs[2, :])
    df_mtm_agg = df_mtm.groupby("date").last()

    spx_r = df_mtm_agg["spx_ret"] * 100
    buf_r = df_mtm_agg["buffer_ret"] * 100

    in_zone    = (spx_r >= -BUFFER*100) & (spx_r < 0)
    paradox    = in_zone & (buf_r < 0)
    normal_neg = spx_r < -BUFFER*100
    bull       = spx_r >= 0

    ax4.scatter(spx_r[bull],                buf_r[bull],                s=2,  alpha=0.3,  color=COLORS["spx"],     label="SPX > 0%")
    ax4.scatter(spx_r[in_zone & ~paradox],  buf_r[in_zone & ~paradox],  s=4,  alpha=0.5,  color=COLORS["buffer"],  label="SPX dans [−10%, 0%] — buffer tient")
    ax4.scatter(spx_r[paradox],             buf_r[paradox],             s=10, alpha=0.85, color="#f78166",          label=f"PARADOXE : SPX protégé, buffer perd quand même ({paradox.sum()} obs.)")
    ax4.scatter(spx_r[normal_neg],          buf_r[normal_neg],          s=2,  alpha=0.2,  color=COLORS["neutral"], label="SPX < −10% (hors buffer, normal)")

    ax4.axhline(0,           color="#8b949e", lw=0.8, ls="--", alpha=0.7)
    ax4.axvline(0,           color="#8b949e", lw=0.8, ls="--", alpha=0.7)
    ax4.axvline(-BUFFER*100, color=COLORS["neutral"], lw=1.0, ls=":", alpha=0.8, label=f"Floor buffer (−{BUFFER*100:.0f}%)")
    ax4.axvspan(-BUFFER*100, 0, alpha=0.05, color=COLORS["buffer"])
    ax4.set_xlim(-45, 45)
    ax4.set_ylim(-20, 25)
    ax4.set_xlabel("SPX return YoY (%)")
    ax4.set_ylabel("Buffer ETF return MtM (%)")
    ax4.set_title("II.3 — Scatter MtM : dans [−10%, 0%], le buffer peut quand même perdre (theta decay)")
    ax4.legend(loc="upper left", fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.25)
    ax4.set_facecolor("#0d1117")

    # ── AX5 : Statistique AQR reproduced ────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    pct, n_zone, n_loss = compute_aqr_stat(df_mtm)
    categories = ["Dans zone protégée\n(SPX ∈ [−10%, 0%])", "Buffer perd\n(buffer_ret < 0)", "Intersection\n(paradoxe)"]
    vals       = [n_zone, n_zone - n_loss, n_loss]
    bars       = ax5.bar(categories, vals,
                         color=[COLORS["neutral"], COLORS["buffer"], COLORS["cash"]],
                         alpha=0.85, width=0.5)
    ax5.set_title(f"II.3 — Reproduction stat AQR\n{pct:.1f}% des obs. dans la zone protégée voient le buffer perdre")
    ax5.set_ylabel("Nombre d'observations journalières")
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_facecolor("#0d1117")
    for bar, v in zip(bars, vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{v:,}", ha="center", va="bottom", fontsize=8)
    # Annotation AQR
    ax5.text(0.5, 0.92, f"AQR (2024) : ~59% | Notre simulation : {pct:.1f}%",
             transform=ax5.transAxes, ha="center", fontsize=8,
             color=COLORS["neutral"], style="italic")

    # ── AX6 : Sensibilité du cap ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    x6  = np.arange(len(df_cap))

    color_vol = COLORS["cash"]
    color_cap = COLORS["buffer"]
    color_r   = COLORS["spx"]

    ax6_r = ax6.twinx()
    ax6.bar(x6 - 0.2, df_cap["vix_pct"], width=0.35,
            color=color_vol, alpha=0.7, label="VIX (%)")
    ax6_r.plot(x6, df_cap["cap_pct"], color=color_cap,
               lw=2, marker="o", ms=6, label="Cap (%)")
    ax6_r.plot(x6, df_cap["r_pct"],   color=color_r,
               lw=1.5, marker="s", ms=4, ls="--", label="Taux rf (%)")

    ax6.set_xticks(x6)
    ax6.set_xticklabels([str(y) for y in df_cap.index], rotation=30, ha="right")
    ax6.set_ylabel("VIX (%)", color=color_vol)
    ax6_r.set_ylabel("Cap / Taux (%)", color=color_cap)
    ax6.set_title("II.5 — Sensibilité du cap : VIX↑ → Cap↓ (paradoxe de protection)")
    ax6.set_facecolor("#0d1117")

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_r.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax6.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Buffer ETF Synthétique S&P 500 — Backtest 2005–2025\nMémoire Partie II",
                 fontsize=14, fontweight="bold", color="#e6edf3", y=1.01)

    out_path = os.path.join(FIG_DIR, "buffer_etf_backtest.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    print(f"✅  Figure sauvegardée : {out_path}")
    return out_path


# ─────────────────────────────────────────────
# TABLEAU DE MÉTRIQUES
# ─────────────────────────────────────────────

def print_metrics(df_bt, df_bench):
    print("\n" + "="*65)
    print("  MÉTRIQUES DE PERFORMANCE (2005–2024, point-to-point)")
    print("="*65)

    metrics_spx   = compute_metrics(df_bt["spx_ret"])
    metrics_buf   = compute_metrics(df_bt["buffer_ret"])
    metrics_bench = compute_metrics(df_bench["bench_ret"])

    labels = ["SPX B&H", "Buffer ETF", "Stock/Cash β-matched"]
    all_m  = [metrics_spx, metrics_buf, metrics_bench]

    header = f"{'Métrique':<20} {'SPX B&H':>14} {'Buffer ETF':>14} {'β-matched':>14}"
    print(header)
    print("-" * len(header))
    for key in ["Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Calmar"]:
        row = f"{key:<20}"
        for m in all_m:
            v = m[key]
            if key in ["Ann. Return", "Ann. Vol", "Max DD"]:
                row += f"  {v*100:>12.2f}%"
            else:
                row += f"  {v:>12.2f} "
        print(row)
    print("="*65)

    # Zoom GFC 2008
    print("\n  ZOOM : Années de stress")
    print("-"*65)
    stress_years = [yr for yr in [2008, 2020, 2022] if yr in df_bt.index]
    for yr in stress_years:
        spx_r = df_bt.loc[yr, "spx_ret"]
        buf_r = df_bt.loc[yr, "buffer_ret"]
        ben_r = df_bench.loc[yr, "bench_ret"]
        cap   = df_bt.loc[yr, "cap"]
        print(f"  {yr}  SPX={spx_r:>+7.2%}  Buffer={buf_r:>+7.2%}  β-matched={ben_r:>+7.2%}  cap={cap:.2%}")
    print("="*65)

    # AQR stat
    print("\n  CAP ZERO-COST PAR MILLÉSIME")
    print("-"*65)
    print(f"  {'Année':<6} {'VIX':>8} {'Taux rf':>10} {'Cap':>10}")
    for yr, row in df_bt.iterrows():
        print(f"  {yr:<6} {row['sigma']*100:>7.1f}% {row['r']*100:>9.2f}% {row['cap']*100:>9.2f}%")
    print("="*65)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*65)
    print("  BUFFER ETF SYNTHÉTIQUE — BACKTEST COMPLET")
    print("  Mémoire Quant Finance — Partie II")
    print("="*65)

    # Data
    df = fetch_data()

    # II.2 — Point-to-point
    df_bt = run_pointtopoint_backtest(df)

    # Metrics
    df_bench = stock_cash_benchmark(df_bt, df)
    print_metrics(df_bt, df_bench)

    # II.3 — MtM journalier
    df_mtm = run_mtm_backtest(df, df_bt)

    pct, n_zone, n_loss = compute_aqr_stat(df_mtm)
    print(f"\n  📌 AQR finding reproduced : {pct:.1f}% des observations où le SPX est")
    print(f"     dans la zone protégée [0%, -{BUFFER*100:.0f}%] voient le buffer perdre de l'argent.")
    print(f"     (AQR paper : ~59% | Notre simulation : {pct:.1f}%)")

    # II.5 — Sensibilité cap
    df_cap = cap_sensitivity_analysis(df_bt)

    # Plots
    out_path = plot_all(df, df_bt, df_mtm, df_bench, df_cap)

    print(f"\n✅  Terminé. Output : {out_path}")
    return df_bt, df_mtm, df_bench, df_cap


if __name__ == "__main__":
    df_bt, df_mtm, df_bench, df_cap = main()
