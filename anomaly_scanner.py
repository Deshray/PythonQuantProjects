"""
anomaly_scanner.py
==================
Statistical Anomaly Detection for Trading Prices
UWaterloo BMath / Data Science

Detects two classes of anomaly in a price series:

  CLASS 1 — RETURN ANOMALIES
    A return r_t is anomalous if it is statistically unlikely under the
    null hypothesis that returns are drawn from the recent empirical
    distribution.  We use two tests:

      (a) Z-score test (normal null):
              z_t = (r_t − μ̂) / σ̂,  two-sided p = 2·Φ(−|z_t|)
          where μ̂, σ̂ are rolling estimates.  Fast but assumes normality.

      (b) t-test (heavier tails):
              t_t = (r_t − μ̂) / (σ̂/√W)
          p-value from t(W−1).  More conservative; better for small windows.

  CLASS 2 — STRUCTURAL BREAKS
    A structural break is when the data-generating process itself shifts —
    the mean or variance of returns changes persistently.

      (a) Rolling Chow test:
          Split the trailing window [t−W, t] in half.  Test H₀: μ₁ = μ₂
          using a pooled two-sample t-test.  A small p-value means the
          first and second halves of the window have different means →
          a shift occurred somewhere in the window.

      (b) CUSUM monitor (Brown-Durbin-Evans):
          Fit baseline μ̂, σ̂ on the formation period.
          Define cumulative standardised residuals:
              C_t = Σ_{s=t₀}^{t}  (r_s − μ̂) / σ̂
          C_t drifts away from 0 when the true mean has shifted.
          Flag when |C_t| exceeds the critical boundary  h · √T,
          where T = number of periods monitored and h is the boundary
          parameter (h ≈ 1.36 gives α = 0.05 asymptotically).

All tests are run in real-time (rolling / sequential) — no look-ahead.

Outputs
-------
  • anomaly_results.csv   — per-day DataFrame with all p-values and flags
  • anomaly_plot.png      — 4-panel diagnostic plot
  • console summary       — anomaly clusters and statistics

Dependencies: numpy, pandas, scipy, statsmodels, matplotlib, seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LAYER  (identical fallback as stat_arb_engine)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_series(ticker: str, start: str, end: str) -> pd.Series:
    """Return a daily adjusted close price Series for a single ticker."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.squeeze()
        raw = raw.dropna()
        if len(raw) > 200:
            print(f"[Data] {ticker}: {len(raw)} live trading days")
            return raw
    except Exception:
        pass

    print(f"[Data] Live data unavailable — simulating {ticker}")
    return _simulate_gbm(ticker, start, end)


def _simulate_gbm(ticker: str, start: str, end: str,
                  mu: float = 0.08, sigma: float = 0.18,
                  seed: int = 42) -> pd.Series:
    """
    Geometric Brownian Motion:   log P_t = log P_{t-1} + μ·dt + σ·√dt·ε_t
    Injects two volatility regimes and one mean-shift to produce detectable anomalies.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, end=end)
    n, dt = len(dates), 1 / 252

    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)

    # ── Inject synthetic anomalies so the scanner has something to find ───
    # Volatility spike around day 400 (lasts 30 days)
    returns[400:430] = np.random.normal(mu * dt, 4 * sigma * np.sqrt(dt), 30)

    # Mean shift around day 700 (lasts 100 days — simulates a regime change)
    returns[700:800] = np.random.normal(-0.20 * dt, sigma * np.sqrt(dt), 100)

    # Single large return at day 250 (flash crash / news shock)
    returns[250] = -0.08   # −8% in one day

    log_prices = np.cumsum(returns)
    prices = 100 * np.exp(log_prices)
    return pd.Series(prices, index=dates, name=ticker)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RETURN ANOMALY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def detect_return_anomalies(prices: pd.Series,
                             window: int = 60,
                             alpha:  float = 0.05) -> pd.DataFrame:
    """
    Rolling return anomaly detection.

    For each day t, fit N(μ̂, σ̂²) on the W returns PRIOR to t (no look-ahead),
    then compute two p-values for r_t:

      p_z  : two-sided z-test   p = 2 · Φ(−|z_t|)          [normal null]
      p_t  : two-sided t-test   p = 2 · P(T_{W-1} ≥ |t_t|)  [t null, heavier tails]

    The t-test is more conservative (wider tails → higher p-values for the
    same |z|), making it less prone to false positives when W is small.

    Parameters
    ----------
    window : int
        Rolling lookback for estimating μ̂ and σ̂.  Default 60 (≈ 3 months).
    alpha  : float
        Significance level for flagging.  Default 0.05.

    Returns
    -------
    DataFrame with columns: return, z_score, p_z, p_t, flag_z, flag_t
    """
    ret = prices.pct_change().dropna()
    results = []

    for i in range(window, len(ret)):
        past = ret.iloc[i - window: i].values
        r_t  = ret.iloc[i]

        mu_hat  = past.mean()
        sig_hat = past.std(ddof=1)

        if sig_hat < 1e-10:
            results.append({
                "date": ret.index[i], "return": r_t,
                "z_score": np.nan, "p_z": np.nan,
                "t_stat": np.nan,  "p_t": np.nan,
                "flag_z": False,   "flag_t": False,
            })
            continue

        z = (r_t - mu_hat) / sig_hat
        t = (r_t - mu_hat) / (sig_hat / np.sqrt(window))

        p_z = 2 * stats.norm.sf(abs(z))
        p_t = 2 * stats.t.sf(abs(t), df=window - 1)

        results.append({
            "date":    ret.index[i],
            "return":  r_t,
            "z_score": z,
            "p_z":     p_z,
            "t_stat":  t,
            "p_t":     p_t,
            "flag_z":  p_z < alpha,
            "flag_t":  p_t < alpha,
        })

    df = pd.DataFrame(results).set_index("date")
    n_z = df["flag_z"].sum()
    n_t = df["flag_t"].sum()
    print(f"[Return Anomalies]  z-test flags: {n_z}  |  t-test flags: {n_t}"
          f"  |  out of {len(df)} days  (α={alpha})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STRUCTURAL BREAK TESTS
# ─────────────────────────────────────────────────────────────────────────────

def detect_chow_breaks(prices: pd.Series,
                        window: int = 60,
                        alpha:  float = 0.05) -> pd.DataFrame:
    """
    Rolling Chow test for a structural break in the mean of returns.

    At each day t, take the trailing W returns and split them in half:
        Left  window:  r[t−W  : t−W/2]   (older half)
        Right window:  r[t−W/2: t]        (recent half)

    H₀: μ_left = μ_right  (no structural shift in the window)
    H₁: μ_left ≠ μ_right  (a shift occurred somewhere in the window)

    We use a two-sample t-test with pooled variance (Welch's variant, which
    does NOT assume equal variances, so it also catches variance breaks).

    The p-value is the probability of observing a t-statistic at least as
    extreme as observed, IF H₀ were true.  A small p-value is evidence for
    a structural break.

    NOTE: this is a rolling test — it fires whenever any part of the trailing
    window contains a break.  The break is localised to roughly the midpoint
    of the window when the flag first appears.

    Returns
    -------
    DataFrame with columns: t_stat, p_chow, flag_chow, delta_mean, delta_vol
    """
    ret = prices.pct_change().dropna()
    half = window // 2
    results = []

    for i in range(window, len(ret)):
        left  = ret.iloc[i - window: i - half].values
        right = ret.iloc[i - half: i].values

        t_stat, p_val = stats.ttest_ind(left, right, equal_var=False)

        results.append({
            "date":       ret.index[i],
            "t_stat":     t_stat,
            "p_chow":     p_val,
            "flag_chow":  p_val < alpha,
            "delta_mean": right.mean() - left.mean(),   # direction of shift
            "delta_vol":  right.std()  - left.std(),    # vol expansion/contraction
        })

    df = pd.DataFrame(results).set_index("date")
    n = df["flag_chow"].sum()
    print(f"[Chow Breaks]       flags: {n}  |  out of {len(df)} days  (α={alpha})")
    return df


def detect_cusum(prices: pd.Series,
                 formation_days: int = 120,
                 boundary_h: float = 1.36,
                 alpha: float = 0.05) -> pd.DataFrame:
    """
    CUSUM (Cumulative Sum) structural break monitor.

    Based on Brown, Durbin & Evans (1975).

    Setup
    ─────
    Fit a baseline mean μ̂ and std σ̂ on the formation period (first
    `formation_days` returns).  These are treated as the "in-control" parameters.

    Monitor
    ───────
    For each post-formation day t, accumulate standardised residuals:

        e_t = (r_t − μ̂) / σ̂          ← how many baseline σ away from baseline μ

        C_t = Σ_{s=t₀}^{t} e_s        ← cumulative sum of residuals

    If the process is still in-control (H₀ true), C_t ≈ 0 (residuals cancel).
    If there is a mean shift Δ, C_t drifts at rate ≈ Δ/σ · T → systematic drift.

    Critical boundary
    ─────────────────
        |C_t| > h · √(T − t₀)

    where T − t₀ is the number of periods monitored and h is a scaling
    constant chosen to control the false-alarm rate:
        h = 1.36  →  α ≈ 0.05  (Kolmogorov-Smirnov asymptotic result)
        h = 1.63  →  α ≈ 0.01

    Interpretation
    ──────────────
        C_t > 0 and drifting up   →  positive mean shift (price trending up)
        C_t < 0 and drifting down →  negative mean shift (price trending down)
        |C_t| crosses boundary    →  structural break detected

    Returns
    -------
    DataFrame with columns: cusum, boundary, flag_cusum, norm_cusum
      norm_cusum = C_t / (h · √T)  ∈ [−1, 1] if in-control; |·| > 1 if break
    """
    ret = prices.pct_change().dropna()

    formation = ret.iloc[:formation_days].values
    mu_hat  = formation.mean()
    sig_hat = formation.std(ddof=1)

    post    = ret.iloc[formation_days:]
    results = []
    cusum   = 0.0

    for j, (date, r) in enumerate(post.items()):
        e      = (r - mu_hat) / sig_hat if sig_hat > 1e-10 else 0.0
        cusum += e
        T_mon  = j + 1                       # periods monitored so far
        bound  = boundary_h * np.sqrt(T_mon)

        results.append({
            "date":        date,
            "cusum":       cusum,
            "boundary":    bound,
            "flag_cusum":  abs(cusum) > bound,
            "norm_cusum":  cusum / bound if bound > 0 else 0.0,
        })

    df = pd.DataFrame(results).set_index("date")
    n = df["flag_cusum"].sum()
    print(f"[CUSUM Monitor]     flags: {n}  |  out of {len(df)} days  "
          f"(h={boundary_h}, α≈{alpha})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMPOSITE SCANNER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyReport:
    ticker:      str
    prices:      pd.Series
    returns:     pd.DataFrame     # from detect_return_anomalies
    chow:        pd.DataFrame     # from detect_chow_breaks
    cusum:       pd.DataFrame     # from detect_cusum
    combined:    pd.DataFrame     # merged view
    summary:     dict = field(default_factory=dict)


def run_scanner(prices: pd.Series,
                window:         int   = 60,
                formation_days: int   = 120,
                alpha:          float = 0.05,
                boundary_h:     float = 1.36) -> AnomalyReport:
    """
    Run all three anomaly tests and merge into a single DataFrame.

    Combined anomaly score
    ──────────────────────
    We define a composite flag that fires when AT LEAST TWO of the three
    tests flag the same day.  This reduces false positives: a single test
    firing on a given day could be a false alarm (~5% of days by construction
    at α=0.05), but two tests firing simultaneously is far less likely under
    the joint null.

    Under independence: P(≥2 of 3 flags | H₀) ≈ 3·(0.05)²·(0.95) + (0.05)³ ≈ 0.7%
    vs P(any single flag | H₀) = 5%.  Requiring agreement is a simple Bonferroni-style guard.
    """
    ticker = prices.name or "ASSET"
    print(f"\n{'═'*55}")
    print(f"  Anomaly Scanner — {ticker}")
    print(f"{'═'*55}")

    ret_df  = detect_return_anomalies(prices, window=window, alpha=alpha)
    chow_df = detect_chow_breaks(prices, window=window, alpha=alpha)
    cusum_df = detect_cusum(prices, formation_days=formation_days,
                            boundary_h=boundary_h, alpha=alpha)

    # ── Merge on date (inner join — all tests must have a value) ──────────
    combined = (ret_df[["return", "z_score", "p_z", "p_t", "flag_z", "flag_t"]]
                .join(chow_df[["p_chow", "flag_chow", "delta_mean", "delta_vol"]],
                      how="inner")
                .join(cusum_df[["cusum", "boundary", "flag_cusum", "norm_cusum"]],
                      how="inner"))

    # Number of tests firing each day
    combined["n_flags"] = (combined["flag_z"].astype(int)
                         + combined["flag_chow"].astype(int)
                         + combined["flag_cusum"].astype(int))

    # Composite: ≥2 tests agree
    combined["flag_composite"] = combined["n_flags"] >= 2

    # Severity: min p-value across the two return tests (CUSUM uses boundary ratio)
    combined["min_pval"] = combined[["p_z", "p_t", "p_chow"]].min(axis=1)

    # ── Summary stats ─────────────────────────────────────────────────────
    n_days = len(combined)
    summary = {
        "ticker":            ticker,
        "days_analysed":     n_days,
        "return_flags":      int(combined["flag_z"].sum()),
        "chow_flags":        int(combined["flag_chow"].sum()),
        "cusum_flags":       int(combined["flag_cusum"].sum()),
        "composite_flags":   int(combined["flag_composite"].sum()),
        "composite_rate_%":  round(combined["flag_composite"].mean() * 100, 2),
        "most_extreme_day":  str(combined["min_pval"].idxmin()),
        "most_extreme_pval": round(float(combined["min_pval"].min()), 6),
    }

    _print_summary(summary)

    return AnomalyReport(
        ticker=ticker, prices=prices,
        returns=ret_df, chow=chow_df, cusum=cusum_df,
        combined=combined, summary=summary,
    )


def _print_summary(s: dict) -> None:
    w = 55
    print(f"\n{'─'*w}")
    print(f"  ANOMALY SUMMARY")
    print(f"{'─'*w}")
    print(f"  Days analysed:       {s['days_analysed']}")
    print(f"  Return flags  (z):   {s['return_flags']}")
    print(f"  Chow break flags:    {s['chow_flags']}")
    print(f"  CUSUM break flags:   {s['cusum_flags']}")
    print(f"  ── Composite (≥2):   {s['composite_flags']}  "
          f"({s['composite_rate_%']}% of days)")
    print(f"  Most extreme day:    {s['most_extreme_day']}")
    print(f"  Minimum p-value:     {s['most_extreme_pval']:.2e}")
    print(f"{'─'*w}")

    # ── Trading interpretation ────────────────────────────────────────────
    rate = s["composite_rate_%"]
    print(f"\n  INTERPRETATION")
    if rate < 1.0:
        print("  ✓  Anomaly rate < 1% — process is mostly well-behaved.")
        print("     Composite flags are likely genuine events (news, macro).")
        print("     USE: enter contrarian trades after return anomalies;")
        print("          reduce position size on CUSUM breaks.")
    elif rate < 3.0:
        print("  ⚠  Anomaly rate 1–3% — moderate instability.")
        print("     Structural breaks suggest regime changes are occurring.")
        print("     USE: flag composite days as high-risk; widen stop-losses;")
        print("          re-estimate hedge ratios after CUSUM boundary breach.")
    else:
        print("  ✗  Anomaly rate > 3% — process is highly non-stationary.")
        print("     Any strategy fitted on historical parameters is likely")
        print("     operating in the wrong regime.  Consider pausing trading")
        print("     until CUSUM reverts inside the boundary.")
    print(f"{'─'*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLUSTER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def find_anomaly_clusters(combined: pd.DataFrame,
                           min_gap_days: int = 5) -> pd.DataFrame:
    """
    Group consecutive anomaly flags into clusters.

    Two anomaly days belong to the same cluster if they are within
    `min_gap_days` of each other.  Returns a DataFrame with cluster
    start/end dates, duration, and peak severity (min p-value).

    Useful for:
      - Risk alerts: a cluster is a sustained regime change, not just noise
      - Signal generation: enter on the FIRST day of a return-anomaly cluster,
        close when the cluster ends (spread has reverted or regime has settled)
    """
    flags = combined[combined["flag_composite"]].copy()
    if flags.empty:
        print("[Clusters] No composite anomaly clusters found.")
        return pd.DataFrame()

    flag_dates = flags.index.tolist()
    clusters   = []
    start      = flag_dates[0]
    prev       = flag_dates[0]

    for d in flag_dates[1:]:
        gap = (d - prev).days
        if gap > min_gap_days:
            clusters.append({"start": start, "end": prev,
                              "duration_days": (prev - start).days + 1})
            start = d
        prev = d
    clusters.append({"start": start, "end": prev,
                     "duration_days": (prev - start).days + 1})

    cluster_df = pd.DataFrame(clusters)

    # Add peak severity (minimum p-value in each cluster)
    peak_pvals = []
    for _, row in cluster_df.iterrows():
        window_data = combined.loc[row["start"]: row["end"], "min_pval"]
        peak_pvals.append(window_data.min())
    cluster_df["peak_pval"] = peak_pvals

    print(f"[Clusters]  {len(cluster_df)} anomaly clusters found:")
    print(cluster_df.to_string(index=False))
    return cluster_df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomalies(report: AnomalyReport,
                   alpha: float = 0.05,
                   save_path: Optional[str] = None) -> None:
    """
    4-panel diagnostic plot.

    Panel 1 — Price series with composite anomaly markers
    Panel 2 — Daily returns with z-score shading and anomaly flags
    Panel 3 — Rolling p-values (log scale) for both return tests + Chow
    Panel 4 — CUSUM with critical boundary
    """
    sns.set_theme(style="darkgrid")
    comb   = report.combined
    prices = report.prices.loc[comb.index]

    fig = plt.figure(figsize=(15, 16))
    fig.suptitle(
        f"Statistical Anomaly Scanner  ·  {report.ticker}\n"
        f"Composite flags: {report.summary['composite_flags']} days "
        f"({report.summary['composite_rate_%']}%)  |  "
        f"Most extreme: {report.summary['most_extreme_day']}  "
        f"(p={report.summary['most_extreme_pval']:.2e})",
        fontsize=13, fontweight="bold", y=0.99)

    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.50)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]

    composite_dates = comb.index[comb["flag_composite"]]

    # ── Panel 1: Price + anomaly markers ─────────────────────────────────
    axs[0].set_title("Price Series  with Composite Anomaly Markers", fontsize=10)
    axs[0].plot(prices.index, prices.values, lw=0.9, color="#4878d0", label=report.ticker)
    if len(composite_dates):
        axs[0].scatter(composite_dates,
                       prices.loc[composite_dates],
                       color="firebrick", s=25, zorder=5,
                       label=f"Composite anomaly (≥2 tests)")
    axs[0].set_ylabel("Price ($)")
    axs[0].legend(fontsize=8)

    # ── Panel 2: Returns + z-score shading ───────────────────────────────
    axs[1].set_title("Daily Returns  with Z-Score Shading", fontsize=10)
    ret = comb["return"]
    axs[1].bar(ret.index, ret.values, width=1.0,
               color=["firebrick" if comb.loc[d, "flag_z"] else "#4878d0"
                      for d in ret.index],
               alpha=0.7, label="Return (red = flagged)")
    mu_roll = ret.rolling(60).mean()
    s_roll  = ret.rolling(60).std()
    axs[1].plot(mu_roll.index, mu_roll + 2*s_roll, lw=0.8, ls="--",
                color="orange", label="±2σ band")
    axs[1].plot(mu_roll.index, mu_roll - 2*s_roll, lw=0.8, ls="--", color="orange")
    axs[1].set_ylabel("Return")
    axs[1].legend(fontsize=8)

    # ── Panel 3: p-values (log scale) ────────────────────────────────────
    axs[2].set_title("Rolling P-Values  (log scale)  —  all three tests", fontsize=10)
    axs[2].semilogy(comb.index, comb["p_z"],    lw=0.8, color="#4878d0",
                    label="Z-test (return)", alpha=0.8)
    axs[2].semilogy(comb.index, comb["p_t"],    lw=0.8, color="#9467bd",
                    label="T-test (return)", alpha=0.8)
    axs[2].semilogy(comb.index, comb["p_chow"], lw=0.8, color="#e377c2",
                    label="Chow (break)", alpha=0.8)
    axs[2].axhline(alpha, color="firebrick", lw=1.2, ls="--",
                   label=f"α = {alpha}")
    axs[2].set_ylabel("p-value (log)")
    axs[2].set_ylim(1e-6, 1.5)
    axs[2].legend(fontsize=8, ncol=4)

    # ── Panel 4: CUSUM + boundary ─────────────────────────────────────────
    axs[3].set_title("CUSUM Monitor  with Critical Boundary  (h = 1.36, α ≈ 0.05)",
                     fontsize=10)
    cusum_data = report.cusum.loc[comb.index]  # align to common index
    axs[3].plot(cusum_data.index, cusum_data["cusum"],
                lw=1.0, color="#2ca02c", label="CUSUM")
    axs[3].plot(cusum_data.index,  cusum_data["boundary"],
                lw=1.0, ls="--", color="firebrick", label="Upper boundary")
    axs[3].plot(cusum_data.index, -cusum_data["boundary"],
                lw=1.0, ls="--", color="firebrick", label="Lower boundary")
    axs[3].fill_between(cusum_data.index,
                        cusum_data["cusum"], cusum_data["boundary"],
                        where=(cusum_data["cusum"] > cusum_data["boundary"]),
                        alpha=0.2, color="firebrick", label="Break detected")
    axs[3].fill_between(cusum_data.index,
                        cusum_data["cusum"], -cusum_data["boundary"],
                        where=(cusum_data["cusum"] < -cusum_data["boundary"]),
                        alpha=0.2, color="steelblue")
    axs[3].axhline(0, lw=0.6, color="gray")
    axs[3].set_ylabel("CUSUM  C_t")
    axs[3].legend(fontsize=8, ncol=2)

    for ax in axs:
        ax.set_xlabel("")

    out = save_path or "/home/claude/anomaly_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRADING INTEGRATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_anomaly_signals(combined: pd.DataFrame,
                              mode: str = "contrarian") -> pd.Series:
    """
    Convert anomaly flags into a trading signal series.

    mode = "contrarian"
        On a return anomaly day (large negative return), go LONG (+1).
        On a large positive return anomaly, go SHORT (−1).
        Exit when composite flag clears.

        Rationale: extreme single-day returns in liquid assets are often
        followed by partial reversal (bid-ask bounce, overreaction).

    mode = "risk_off"
        Go FLAT (0) whenever ANY anomaly flag fires.
        Exit flat when all flags clear.

        Rationale: anomalies signal elevated uncertainty; better to sit out
        than to trade with mis-estimated parameters.

    Returns a Series of {−1, 0, +1} aligned to combined.index.
    """
    sig = pd.Series(0, index=combined.index, dtype=float)

    if mode == "contrarian":
        for date, row in combined.iterrows():
            if row["flag_z"] and not row["flag_cusum"]:
                # Return anomaly but NOT a structural break
                # → likely a transient shock, fade it
                sig.loc[date] = +1 if row["return"] < 0 else -1

    elif mode == "risk_off":
        for date, row in combined.iterrows():
            if row["flag_composite"]:
                sig.loc[date] = 0   # already 0, but makes intent explicit
            # Otherwise leave as 0 — this signal gates a primary strategy

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'contrarian' or 'risk_off'.")

    n_long  = (sig == +1).sum()
    n_short = (sig == -1).sum()
    n_flat  = (sig ==  0).sum()
    print(f"[Signals] mode={mode}  long={n_long}  short={n_short}  flat={n_flat}")
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────
    TICKER          = "SPY"          # try any ticker; falls back to synthetic
    START, END      = "2018-01-01", "2024-01-01"
    WINDOW          = 60             # rolling window for return tests & Chow
    FORMATION_DAYS  = 120            # baseline period for CUSUM
    ALPHA           = 0.05           # significance threshold
    BOUNDARY_H      = 1.36           # CUSUM boundary parameter (α ≈ 0.05)

    # ── 1. Data ───────────────────────────────────────────────────────────
    prices = fetch_price_series(TICKER, START, END)

    # ── 2. Run all tests ──────────────────────────────────────────────────
    report = run_scanner(
        prices,
        window=WINDOW,
        formation_days=FORMATION_DAYS,
        alpha=ALPHA,
        boundary_h=BOUNDARY_H,
    )

    # ── 3. Cluster analysis ───────────────────────────────────────────────
    clusters = find_anomaly_clusters(report.combined, min_gap_days=5)

    # ── 4. Generate trading signals ───────────────────────────────────────
    contrarian_signals = generate_anomaly_signals(report.combined, mode="contrarian")
    risk_off_signals   = generate_anomaly_signals(report.combined, mode="risk_off")

    # ── 5. Save results ───────────────────────────────────────────────────
    report.combined.to_csv("/home/claude/anomaly_results.csv")
    print("[Data] Full results saved → anomaly_results.csv")

    # ── 6. Plot ───────────────────────────────────────────────────────────
    plot_anomalies(report, alpha=ALPHA)
