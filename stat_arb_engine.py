"""
stat_arb_engine.py
==================
Statistical Arbitrage (Pairs Trading) Backtesting Engine
BMath / Data Science — University of Waterloo

Key concepts (in order of use):
  1. Cointegration (Engle-Granger):  log(P_A) = β·log(P_B) + OU
  2. Ornstein-Uhlenbeck process:     the spread mean-reverts
  3. Rolling z-score signal:         trade when spread is "too far" from mean
  4. Dollar-neutral sizing:          equal dollar value in each leg
  5. Walk-forward backtest:          estimate β on formation data, trade out-of-sample
  6. Performance metrics:            Sharpe, Sortino, Calmar, Max Drawdown, CAGR

Dependencies: statsmodels, pandas, numpy, matplotlib, seaborn
Optional:     yfinance (synthetic data used if unavailable)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted close prices for a list of tickers.
    Falls back to a synthetic simulation if live data is unavailable.
    """
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        raw = raw.dropna(axis=1, thresh=int(0.95 * len(raw))).ffill().dropna()
        if raw.shape[0] > 200 and raw.shape[1] >= 2:
            print(f"[Data] {raw.shape[1]} live tickers × {raw.shape[0]} trading days")
            return raw
    except Exception:
        pass

    print("[Data] Live data unavailable — using synthetic market simulation")
    return _simulate_universe(tickers, start, end)


def _simulate_universe(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Synthetic market universe with cointegrated pairs in LOG-PRICE SPACE.

    For each adjacent pair (A, B):

        log P_B(t) = GBM                                 ← I(1) process
        log P_A(t) = β · log P_B(t) + S(t)              ← same I(1) trend + OU residual

    where S(t) is an Ornstein-Uhlenbeck process:

        ΔS_t = −θ · S_{t-1} + σ_OU · ε_t,  ε_t ~ N(0,1)

    This construction guarantees:
      (a) Both series are I(1)  [random walk with drift]
      (b) The log-price spread log(P_A) − β·log(P_B) = S(t) is I(0)  [stationary]
      (c) OLS of log(P_A) on log(P_B) recovers β unbiasedly (in population)
    """
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n, dt = len(dates), 1 / 252
    df    = pd.DataFrame(index=dates)

    for i in range(0, len(tickers) - 1, 2):
        t_a, t_b = tickers[i], tickers[i + 1]

        # ── Base asset: GBM in log-price space ──────────────────────────
        drift_log = 0.08 * dt                          # ~8% annual
        vol_log   = 0.18 * np.sqrt(dt)                # ~18% annual vol
        log_PB    = np.cumsum(np.random.normal(drift_log, vol_log, n))

        # ── True hedge ratio ─────────────────────────────────────────────
        beta_true = round(np.random.uniform(0.8, 1.4), 3)

        # ── OU spread in LOG space ───────────────────────────────────────
        #   Half-life ∈ [15, 35] days → clear, tradeable mean-reversion
        half_life = np.random.uniform(15, 35)
        theta     = np.log(2) / half_life              # daily mean-reversion speed

        #   sig_ou calibrated so spread std ≈ 3%–8% of price (in log units)
        sig_ou    = np.random.uniform(0.015, 0.025)

        S = np.zeros(n)
        for j in range(1, n):
            S[j] = S[j-1] * (1 - theta) + sig_ou * np.random.randn()

        # ── Construct log P_A = β·log P_B + OU ──────────────────────────
        log_PA = beta_true * log_PB + S

        df[t_b] = 100 * np.exp(log_PB)
        df[t_a] = 100 * np.exp(log_PA)

    # Odd ticker: standalone GBM
    if len(tickers) % 2 == 1:
        t = tickers[-1]
        df[t] = 100 * np.exp(np.cumsum(
            np.random.normal(0.08*dt, 0.18*np.sqrt(dt), n)))

    print(f"[Data] {df.shape[1]} synthetic tickers × {df.shape[0]} trading days")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COINTEGRATION SCREENING  (in log-price space)
# ─────────────────────────────────────────────────────────────────────────────

def engle_granger_matrix(prices: pd.DataFrame,
                         pvalue_cutoff: float = 0.05,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Screen all ticker pairs for cointegration using the Engle-Granger test.

    WHY LOG PRICES?
    ───────────────
    For equities, the economically natural cointegrating relationship is:
        log P_A = α + β · log P_B + ε_t

    This captures the ratio relationship P_A / P_B^β ≈ const, which is
    far more stable than a level relationship P_A − β·P_B ≈ const.

    The EG test:
      Step 1 — OLS: log P_A = α + β·log P_B  →  get residuals ε̂_t
      Step 2 — ADF on ε̂_t: reject H₀ (unit root) ⟺ spread is I(0)

    Returns
    -------
    DataFrame with ticker_1, ticker_2, pvalue, hedge_ratio (β̂),
    sorted ascending by pvalue, filtered to pvalue < pvalue_cutoff.
    """
    log_prices = np.log(prices)
    tickers    = prices.columns.tolist()
    results    = []

    for t1, t2 in combinations(tickers, 2):
        y = log_prices[t1].values
        x = add_constant(log_prices[t2].values)

        # OLS gives a super-consistent estimator of the cointegrating vector
        beta = OLS(y, x).fit().params[1]

        # EG joint test (MacKinnon critical values)
        _, pval, _ = coint(log_prices[t1], log_prices[t2])

        results.append({"ticker_1": t1, "ticker_2": t2,
                        "pvalue": round(pval, 4), "hedge_ratio": round(beta, 4)})

    df = (pd.DataFrame(results)
            .sort_values("pvalue")
            .query("pvalue < @pvalue_cutoff")
            .reset_index(drop=True))

    if verbose:
        print(f"\n[Cointegration] {len(df)} cointegrated pairs "
              f"(p < {pvalue_cutoff}) out of {len(results)} tested\n")
        if not df.empty:
            print(df.to_string(index=False))
    return df


def adf_summary(log_spread: pd.Series, label: str = "Spread") -> None:
    """Augmented Dickey-Fuller test on a (log-price) spread."""
    stat, pval, _, nobs, crit, _ = adfuller(log_spread.dropna(), autolag="AIC")
    result = "✓ stationary (I(0))" if pval < 0.05 else "✗ NOT stationary"
    print(f"[ADF] {label}: stat={stat:.3f}, p={pval:.4f}  →  {result}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOG-PRICE SPREAD & Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_spread(prices: pd.DataFrame, t1: str, t2: str,
                       hedge_ratio: float) -> pd.Series:
    """
    Log-price spread:

        spread_t = log P_{t1}  −  β̂ · log P_{t2}

    This is the Engle-Granger residual — the component of log P_{t1}
    that is NOT explained by the common stochastic trend β̂·log P_{t2}.
    By construction, if the pair is cointegrated, this is an I(0) (stationary)
    process that mean-reverts.
    """
    return np.log(prices[t1]) - hedge_ratio * np.log(prices[t2])


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score:

        z_t  =  ( spread_t  −  μ̂_t )  /  σ̂_t

    μ̂_t, σ̂_t are the sample mean and standard deviation over
    the trailing `window` observations.

    Intuition: z_t measures how many rolling standard deviations the spread
    has deviated from its recent mean.  The OU process implies that large
    positive or negative z-scores are followed by mean reversion toward 0.
    """
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    return (spread - mu) / sig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(zscore: pd.Series,
                     entry_z: float = 2.0,
                     exit_z:  float = 0.5) -> pd.Series:
    """
    Mean-reversion signal from the z-score (state machine):

      z_t < −entry_z  →  open  LONG  spread (+1):  bet spread rises to 0
      z_t > +entry_z  →  open  SHORT spread (−1):  bet spread falls to 0
      |z_t| < exit_z  →  FLAT (0):                 spread has reverted, take profit

    The state machine prevents double-entry and handles exit correctly:
      - A LONG  position is held until z_t ≥ −exit_z  (reversion is near-complete)
      - A SHORT position is held until z_t ≤ +exit_z

    Returns a Series of {−1, 0, +1} with the same index as zscore.
    """
    signals  = pd.Series(0, index=zscore.index, dtype=float)
    position = 0

    for i, z in enumerate(zscore):
        if np.isnan(z):
            continue
        if position == 0:
            if   z < -entry_z: position = +1
            elif z >  entry_z: position = -1
        elif position == +1:
            if z >= -exit_z:   position =  0    # mean-reverted upward
        elif position == -1:
            if z <=  exit_z:   position =  0    # mean-reverted downward

        signals.iloc[i] = position

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    equity_curve:  pd.Series
    daily_returns: pd.Series
    signals:       pd.Series
    log_spread:    pd.Series
    zscore:        pd.Series
    pos_t1:        pd.Series   # signed dollar value of t1 position
    pos_t2:        pd.Series   # signed dollar value of t2 position
    trades:        pd.DataFrame
    metrics:       dict = field(default_factory=dict)


def backtest_pair(prices:           pd.DataFrame,
                  t1:               str,
                  t2:               str,
                  formation_days:   int   = 252,
                  zscore_window:    int   = 60,
                  entry_z:          float = 2.0,
                  exit_z:           float = 0.5,
                  capital:          float = 100_000,
                  transaction_cost: float = 0.001) -> BacktestResult:
    """
    Walk-forward out-of-sample backtest for a single cointegrated pair.

    Walk-forward structure
    ──────────────────────
    ┌──────────────────────┬──────────────────────────────────┐
    │   Formation period   │  Trading period (out-of-sample)  │
    │   (formation_days)   │  (all remaining data)            │
    │  → OLS hedge ratio β̂ │  → z-score, signals, P&L        │
    └──────────────────────┴──────────────────────────────────┘

    Position sizing  (dollar-neutral)
    ─────────────────────────────────
    At entry, we invest exactly ½ of current portfolio value in each leg:

      Long  spread (+1):  buy  $½ of t1,  sell $½ of t2
      Short spread (−1):  sell $½ of t1,  buy  $½ of t2

    Dollar-neutral ≈ market-neutral: the systematic equity risk of the long
    and short legs approximately cancel (they share the same β to the market).

    P&L accounting
    ──────────────
    CRITICAL ORDER:  mark-to-market FIRST with yesterday's positions,
    THEN rebalance if the signal changed.

    This avoids the common bug of earning returns on positions we just opened.

      1. PnL_t = n_A · (P_A,t − P_A,t−1) + n_B · (P_B,t − P_B,t−1)
      2. portfolio_t = portfolio_{t-1} + PnL_t
      3. if signal changed: close → pay TC, open new → pay TC

    Parameters
    ----------
    formation_days    : days used to fit the OLS hedge ratio
    zscore_window     : rolling window for z-score normalisation
    entry_z / exit_z  : signal thresholds in z-score units
    capital           : starting portfolio value ($)
    transaction_cost  : one-way fractional cost (e.g. 0.001 = 10 bps)
    """
    if len(prices) < formation_days + zscore_window + 30:
        raise ValueError("Insufficient data. Reduce formation_days or zscore_window.")

    form_prices  = prices.iloc[:formation_days]
    trade_prices = prices.iloc[formation_days:].copy()

    # ── Formation: OLS hedge ratio in LOG-PRICE SPACE ─────────────────────
    log_y = np.log(form_prices[t1].values)
    log_x = add_constant(np.log(form_prices[t2].values))
    reg   = OLS(log_y, log_x).fit()
    hedge_ratio = reg.params[1]          # β̂

    # ── Trading: log-spread → z-score → signals ───────────────────────────
    log_spread = compute_log_spread(trade_prices, t1, t2, hedge_ratio)
    zscore     = rolling_zscore(log_spread, zscore_window)
    signals    = generate_signals(zscore, entry_z, exit_z)

    # ── Dollar amounts of each leg per day ────────────────────────────────
    P1 = trade_prices[t1].values
    P2 = trade_prices[t2].values
    idx = signals.index

    portfolio_value = capital
    equity_curve = []
    daily_rets   = []
    p1_log, p2_log = [], []
    trade_log = []

    prev_sig = 0
    na = 0.0    # shares of t1 (+ = long, − = short)
    nb = 0.0    # shares of t2

    for i, (date, sig) in enumerate(signals.items()):
        pa, pb = P1[i], P2[i]

        # ── Step 1: Mark-to-market with YESTERDAY'S positions ────────────
        if i > 0:
            pnl = na * (pa - P1[i - 1]) + nb * (pb - P2[i - 1])
            portfolio_value += pnl

        # ── Step 2: Rebalance if signal changed ───────────────────────────
        if sig != prev_sig:
            # Close existing position
            close_cost = transaction_cost * (abs(na * pa) + abs(nb * pb))
            portfolio_value -= close_cost
            na = nb = 0.0

            # Open new position (dollar-neutral)
            if sig != 0:
                half = portfolio_value / 2.0
                na   =  sig * half / pa   # +shares of t1 if long spread
                nb   = -sig * half / pb   # −shares of t2 if long spread

                open_cost = transaction_cost * (abs(na * pa) + abs(nb * pb))
                portfolio_value -= open_cost

            trade_log.append({"date": date, "signal": sig,
                               "log_spread": log_spread.loc[date],
                               "zscore": zscore.loc[date],
                               "portfolio_value": portfolio_value})

        prev_pv = portfolio_value
        prev_pnl = na * (pa - P1[i-1]) + nb * (pb - P2[i-1]) if i > 0 else 0
        equity_curve.append(portfolio_value)
        prev_ret = prev_pnl / (portfolio_value - prev_pnl) if (portfolio_value - prev_pnl) > 0 else 0
        daily_rets.append(prev_ret)
        p1_log.append(na * pa)
        p2_log.append(nb * pb)
        prev_sig = sig

    equity = pd.Series(equity_curve,  index=idx, name="equity")
    dret   = pd.Series(daily_rets,    index=idx, name="daily_return")
    p1s    = pd.Series(p1_log,        index=idx, name=t1)
    p2s    = pd.Series(p2_log,        index=idx, name=t2)
    trades = pd.DataFrame(trade_log).set_index("date") if trade_log else pd.DataFrame()

    result = BacktestResult(
        equity_curve=equity, daily_returns=dret,
        signals=signals, log_spread=log_spread, zscore=zscore,
        pos_t1=p1s, pos_t2=p2s, trades=trades,
    )
    result.metrics = compute_metrics(result, capital)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def newey_west_sharpe(excess_returns: pd.Series,
                      nlags: int = None,
                      ann_factor: float = 252) -> float:
    """
    Newey-West (HAC) corrected annualised Sharpe ratio.

    WHY THIS MATTERS
    ────────────────
    Stat-arb returns are autocorrelated: while a position is open, each day's
    P&L is mechanically related to the previous day's (same trade, same legs).
    Naive Sharpe treats all returns as i.i.d., which understates variance and
    OVERSTATES the Sharpe ratio — sometimes by 30–50%.

    Newey-West corrects for this by estimating the long-run (HAC) variance:

        Ω_NW  =  γ_0  +  2 · Σ_{l=1}^{L}  w_l · γ_l

    where  γ_l = (1/T) Σ_{t=l+1}^{T} (r_t − μ̄)(r_{t−l} − μ̄)   [autocovariance at lag l]
    and    w_l = 1 − l/(L+1)                                       [Bartlett weights, ensure Ω_NW ≥ 0]

    The NW Sharpe uses Ω_NW directly as the variance proxy:

        SR_NW  =  μ̄_excess · √252  /  √Ω_NW

    Note: Ω_NW estimates the variance of a SINGLE daily return (the long-run
    variance), so annualisation by √252 is correct as for the naive Sharpe.

    The default lag L follows the Newey-West (1994) plug-in rule:
        L = floor( 4 · (T/100)^{2/9} )
    which grows slowly with T (e.g. L ≈ 8 for T = 500, L ≈ 10 for T = 1000).
    """
    r = excess_returns.dropna().values
    T = len(r)
    if T < 10:
        return 0.0

    if nlags is None:
        nlags = int(np.floor(4 * (T / 100) ** (2 / 9)))   # NW 1994 rule

    mu = r.mean()
    e  = r - mu   # demeaned residuals

    # Γ_0: variance term
    long_run_var = np.dot(e, e) / T

    # Bartlett-weighted autocovariance terms
    for l in range(1, nlags + 1):
        w       = 1.0 - l / (nlags + 1)          # Bartlett weight ∈ (0, 1)
        gamma_l = np.dot(e[l:], e[:-l]) / T      # autocovariance at lag l
        long_run_var += 2 * w * gamma_l

    if long_run_var <= 0:
        return 0.0

    return mu * np.sqrt(ann_factor) / np.sqrt(long_run_var)


def compute_metrics(result: BacktestResult, initial_capital: float,
                    risk_free_rate: float = 0.045) -> dict:
    """
    Standard quantitative finance performance metrics.

    Sharpe Ratio (naive)  =  (μ_excess / σ_daily) × √252   [assumes i.i.d. returns]
    Sharpe Ratio (NW)     =  μ_excess × √252 / √Ω_NW        [HAC-corrected; use this]
    Sortino Ratio         =  (μ_excess / σ_downside) × √252 [downside-only vol]
    Max Drawdown          =  min_t [ V_t / max_{s≤t}(V_s) − 1 ]
    CAGR                  =  (V_T / V_0)^{252/T} − 1
    Calmar Ratio          =  CAGR / |Max Drawdown|
    """
    eq  = result.equity_curve
    ret = result.daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

    rf_daily = risk_free_rate / 252
    excess   = ret - rf_daily

    sharpe  = (excess.mean() / excess.std() * np.sqrt(252)
               if excess.std() > 1e-10 else 0.0)
    sharpe_nw = newey_west_sharpe(excess)

    rolling_max = eq.cummax()
    max_dd      = ((eq - rolling_max) / rolling_max).min()

    T    = len(eq)
    cagr = (eq.iloc[-1] / initial_capital) ** (252 / T) - 1

    calmar  = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    neg_exc = excess[ret < rf_daily]
    sortino = (excess.mean() / neg_exc.std() * np.sqrt(252)
               if len(neg_exc) > 1 else 0.0)

    # Win rate: fraction of trade open→close cycles with positive P&L
    tr = result.trades
    win_rate = np.nan
    if len(tr) > 2:
        pv = tr["portfolio_value"].values
        win_rate = np.mean(np.diff(pv) > 0)

    return {
        "Total Return (%)":    round((eq.iloc[-1] / initial_capital - 1) * 100, 2),
        "CAGR (%)":            round(cagr * 100, 2),
        "Sharpe (naive)":      round(sharpe, 3),
        "Sharpe (NW-HAC)":     round(sharpe_nw, 3),
        "Sortino Ratio":       round(sortino, 3),
        "Calmar Ratio":        round(calmar, 3),
        "Max Drawdown (%)":    round(max_dd * 100, 2),
        "Volatility (ann %)":  round(ret.std() * np.sqrt(252) * 100, 2),
        "# Trades":            len(tr),
        "Win Rate (%)":        f"{win_rate*100:.1f}" if not np.isnan(win_rate) else "N/A",
        "Time in Market (%)":  round((result.signals != 0).mean() * 100, 1),
        "Final Value ($)":     round(eq.iloc[-1], 2),
    }


def print_metrics(metrics: dict, pair: tuple) -> None:
    w = 46
    print(f"\n{'═' * w}")
    print(f"  Backtest Results:  {pair[0]} / {pair[1]}")
    print(f"{'═' * w}")
    for k, v in metrics.items():
        print(f"  {k:<30} {str(v):>10}")
    print(f"{'═' * w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest(result: BacktestResult, pair: tuple,
                  initial_capital: float,
                  entry_z: float = 2.0, exit_z: float = 0.5,
                  save_path: Optional[str] = None) -> None:
    """
    4-panel diagnostic figure.

    Panel 1 — Log-price spread with rolling Bollinger bands
    Panel 2 — Z-score with entry/exit thresholds and signal shading
    Panel 3 — Portfolio equity curve with drawdown shading
    Panel 4 — Rolling 60-day annualised Sharpe ratio
    """
    t1, t2   = pair
    m        = result.metrics
    sns.set_theme(style="darkgrid")

    fig = plt.figure(figsize=(15, 14))
    fig.suptitle(
        f"Statistical Arbitrage Backtest  ·  {t1} / {t2}\n"
        f"Sharpe(naive) {m['Sharpe (naive)']}  |  Sharpe(NW) {m['Sharpe (NW-HAC)']}  |  "
        f"CAGR {m['CAGR (%)']}%  |  Max DD {m['Max Drawdown (%)']}%  |  Trades {m['# Trades']}",
        fontsize=13, fontweight="bold", y=0.99)

    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.50)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]

    sp  = result.log_spread
    z   = result.zscore
    sig = result.signals
    eq  = result.equity_curve
    ret = result.daily_returns
    W   = 60

    # ── Panel 1: log-spread ───────────────────────────────────────────────
    rm = sp.rolling(W).mean()
    rs = sp.rolling(W).std()
    axs[0].set_title(f"Log-Price Spread   log(P_{t1}) − β̂·log(P_{t2})", fontsize=10)
    axs[0].plot(sp.index, sp, lw=0.8, color="#4878d0", label="Spread")
    axs[0].plot(rm.index, rm, lw=1.1, ls="--", color="orange", label=f"{W}d mean")
    axs[0].fill_between(rm.index, rm - rs, rm + rs,
                        alpha=0.15, color="orange", label="±1σ band")
    axs[0].set_ylabel("Log-spread")
    axs[0].legend(fontsize=8, loc="upper right")

    # ── Panel 2: z-score + signal shading ────────────────────────────────
    axs[1].set_title("Z-score  with entry / exit thresholds", fontsize=10)
    axs[1].plot(z.index, z.clip(-4, 4), lw=0.8, color="#444", zorder=3, label="Z-score")
    axs[1].axhline( entry_z, color="firebrick",    lw=1.2, ls="--", label=f"+{entry_z}")
    axs[1].axhline(-entry_z, color="steelblue",    lw=1.2, ls="--", label=f"−{entry_z}")
    axs[1].axhline( exit_z,  color="salmon",       lw=0.8, ls=":")
    axs[1].axhline(-exit_z,  color="lightskyblue", lw=0.8, ls=":")
    axs[1].axhline(0,        color="gray",          lw=0.5)
    axs[1].fill_between(sig.index, z.clip(-4,4), 0,
                        where=(sig == +1), alpha=0.22, color="steelblue", label="Long")
    axs[1].fill_between(sig.index, z.clip(-4,4), 0,
                        where=(sig == -1), alpha=0.22, color="firebrick", label="Short")
    axs[1].set_ylim(-4.5, 4.5)
    axs[1].set_ylabel("Z-score")
    axs[1].legend(fontsize=8, ncol=4)

    # ── Panel 3: equity curve + drawdown ─────────────────────────────────
    axs[2].set_title("Portfolio Equity Curve", fontsize=10)
    axs[2].plot(eq.index, eq, lw=1.5, color="#2ca02c", label="Strategy")
    axs[2].axhline(initial_capital, lw=0.8, ls=":", color="gray", label="Initial capital")
    peak = eq.cummax()
    axs[2].fill_between(eq.index, eq, peak,
                        where=(eq < peak), alpha=0.20, color="red", label="Drawdown")
    axs[2].set_ylabel("Portfolio Value ($)")
    axs[2].legend(fontsize=8)

    # ── Panel 4: rolling Sharpe ───────────────────────────────────────────
    axs[3].set_title(f"Rolling {W}-Day Sharpe Ratio  (annualised)", fontsize=10)
    rf_daily = 0.045 / 252
    exc = ret - rf_daily
    rolling_sr = exc.rolling(W).mean() / exc.rolling(W).std() * np.sqrt(252)
    axs[3].plot(rolling_sr.index, rolling_sr, lw=0.9, color="#9467bd")
    axs[3].axhline(0,  lw=0.9, ls="--", color="gray")
    axs[3].axhline(1,  lw=0.7, ls=":",  color="#2ca02c", label="SR = 1")
    axs[3].axhline(-1, lw=0.7, ls=":",  color="firebrick")
    axs[3].set_ylim(-5, 5)
    axs[3].set_ylabel("Sharpe")
    axs[3].legend(fontsize=8)

    for ax in axs:
        ax.set_xlabel("")

    out = save_path or "/home/claude/backtest_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(prices:              pd.DataFrame,
                          t1:                 str,
                          t2:                 str,
                          entry_z_grid:       list = [1.5, 2.0, 2.5],
                          exit_z_grid:        list = [0.25, 0.5, 0.75],
                          zscore_window_grid: list = [30, 60, 90],
                          formation_days:     int   = 252,
                          capital:            float = 100_000) -> pd.DataFrame:
    """
    Grid search over (zscore_window, entry_z, exit_z).

    Only signal parameters vary — the formation window is fixed, so the OLS
    hedge ratio is the same across all combinations.  This avoids overfitting
    the estimation window to the trading period.

    Returns a DataFrame of results sorted by descending Sharpe.
    """
    combos = [(w, ez, xz)
              for w  in zscore_window_grid
              for ez in entry_z_grid
              for xz in exit_z_grid
              if xz < ez]

    print(f"\n[Sensitivity] {len(combos)} valid parameter combinations …")
    rows = []
    for w, ez, xz in combos:
        try:
            r = backtest_pair(prices, t1, t2,
                              formation_days=formation_days,
                              zscore_window=w, entry_z=ez, exit_z=xz,
                              capital=capital)
            m = r.metrics
            rows.append({"window": w, "entry_z": ez, "exit_z": xz,
                         "Sharpe(NW)": m["Sharpe (NW-HAC)"], "Sharpe(naive)": m["Sharpe (naive)"],
                         "CAGR%": m["CAGR (%)"],
                         "MaxDD%": m["Max Drawdown (%)"], "Trades": m["# Trades"]})
        except Exception:
            pass

    df = pd.DataFrame(rows).sort_values("Sharpe(NW)", ascending=False).reset_index(drop=True)
    print(df.head(10).to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN  —  EXAMPLE RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────────
    UNIVERSE = [
        "KO",   "PEP",    # Consumer staples
        "GS",   "MS",     # Investment banks
        "XOM",  "CVX",    # Energy
        "WMT",  "TGT",    # Retail
        "MSFT", "ORCL",   # Technology
    ]

    START, END          = "2018-01-01", "2024-01-01"
    FORMATION_DAYS      = 252 * 3  # 3 years to estimate hedge ratio (spec: train on first 3yr)
    ZSCORE_WINDOW       = 60       # 60-day rolling z-score
    ENTRY_Z, EXIT_Z     = 2.0, 0.5
    INITIAL_CAPITAL     = 100_000
    TRANSACTION_COST    = 0.001    # 10 bps one-way

    print("=" * 60)
    print("  Statistical Arbitrage Backtesting Engine")
    print("  UWaterloo BMath / Data Science")
    print("=" * 60)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    prices = fetch_prices(UNIVERSE, START, END)

    # ── 2. Screen for cointegrated pairs (FORMATION WINDOW ONLY!) ─────────────
    coint_df = engle_granger_matrix(
        prices.iloc[:FORMATION_DAYS], pvalue_cutoff=0.10)

    if coint_df.empty:
        print("[!] No cointegrated pairs found. Widening to p < 0.20 …")
        coint_df = engle_granger_matrix(
            prices.iloc[:FORMATION_DAYS], pvalue_cutoff=0.20)

    if coint_df.empty:
        print("[!] Still no pairs found. Exiting.")
        exit()

    # ── 3. Verify spread stationarity and backtest top pair ───────────────────
    best   = coint_df.iloc[0]
    T1, T2 = best["ticker_1"], best["ticker_2"]
    BETA   = best["hedge_ratio"]

    print(f"\n[Engine] Top pair: {T1} / {T2}  "
          f"(EG p={best['pvalue']}, β̂={BETA})")

    # ADF check on formation spread
    form_log_spread = compute_log_spread(
        prices.iloc[:FORMATION_DAYS], T1, T2, BETA)
    adf_summary(form_log_spread, f"log({T1}) − β̂·log({T2})")

    # ── 4. Backtest ───────────────────────────────────────────────────────────
    result = backtest_pair(
        prices=prices, t1=T1, t2=T2,
        formation_days=FORMATION_DAYS, zscore_window=ZSCORE_WINDOW,
        entry_z=ENTRY_Z, exit_z=EXIT_Z,
        capital=INITIAL_CAPITAL, transaction_cost=TRANSACTION_COST,
    )
    print_metrics(result.metrics, (T1, T2))

    # ── Honest result ─────────────────────────────────────────────────────────
    sr_nw    = result.metrics["Sharpe (NW-HAC)"]
    sr_naive = result.metrics["Sharpe (naive)"]
    autocorr_bias = sr_naive - sr_nw

    print("─" * 60)
    print("  HONEST ASSESSMENT")
    print("─" * 60)
    print(f"  Naive Sharpe:          {sr_naive:>7.3f}  (assumes i.i.d. daily returns)")
    print(f"  Newey-West Sharpe:     {sr_nw:>7.3f}  (HAC-corrected for autocorrelation)")
    print(f"  Autocorrelation bias:  {autocorr_bias:>+7.3f}  (naive − NW; how much was overstated)")
    print()
    if sr_nw < 0.5:
        print("  ⚠  NW Sharpe < 0.5 — strategy does NOT clear a reasonable bar.")
        print()
        print("  Why:")
        print("  1. Transaction costs: each round-trip (open + close) costs ~20 bps.")
        print(f"     With {result.metrics['# Trades']} trades, total drag ≈ "
              f"{result.metrics['# Trades'] * 0.002 * 100:.1f}% on $100k.")
        print("  2. Regime changes: the cointegrating relationship fitted on the")
        print("     formation period may break down in the test window (2022 rate")
        print("     shock, correlation breakdown). The EG test is backward-looking.")
        print("  3. Autocorrelated losses: when a trade moves against us, each day")
        print("     the position is held amplifies the loss. NW Sharpe penalises")
        print("     this; naive Sharpe ignores it.")
        print("  4. Multiple testing: we selected this pair as the lowest EG p-value")
        print("     out of several candidates. In-sample significance does not")
        print("     guarantee out-of-sample mean reversion.")
    elif sr_nw < 1.0:
        print("  ✓  NW Sharpe in [0.5, 1.0) — marginal. Investable only with low")
        print("     execution costs and after further robustness checks.")
    else:
        print("  ✓✓ NW Sharpe ≥ 1.0 — strong result. Verify no look-ahead bias.")
    print("─" * 60)

    # ── 5. Sensitivity analysis ───────────────────────────────────────────────
    sens_df = sensitivity_analysis(
        prices, T1, T2,
        entry_z_grid=[1.5, 2.0, 2.5],
        exit_z_grid=[0.25, 0.5, 0.75],
        zscore_window_grid=[30, 60, 90],
        formation_days=FORMATION_DAYS,
    )

    # ── 6. Visualise ──────────────────────────────────────────────────────────
    plot_backtest(result, (T1, T2), INITIAL_CAPITAL, ENTRY_Z, EXIT_Z)

    print(f"\n[Done]  Sharpe(NW)={result.metrics['Sharpe (NW-HAC)']} "
          f"| Return={result.metrics['Total Return (%)']}% "
          f"| Trades={result.metrics['# Trades']}")
