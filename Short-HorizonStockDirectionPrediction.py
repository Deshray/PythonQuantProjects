import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG",
    "JPM", "BAC", "GS", "MS",
    "TSLA", "AMD", "INTC",
    "XOM", "CVX",
    "UNH", "JNJ",
    "SPY"
]

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
HORIZON = 5           # forward return horizon
LONG_Q = 0.8          # top 20%
SHORT_Q = 0.2         # bottom 20%
TRANSACTION_COST = 0.001


# ------------------------------------------------------------
# Data Download + Panel Format
# ------------------------------------------------------------

def download_price_data(tickers):
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        progress=False
    )
    return data


def to_panel(data):
    panels = []
    for ticker in data.columns.levels[0]:
        df = data[ticker].copy()
        df["ticker"] = ticker
        df = df.reset_index()
        panels.append(df)

    panel = pd.concat(panels, ignore_index=True)
    panel.columns = panel.columns.str.lower()
    return panel


# ------------------------------------------------------------
# Feature Engineering (per ticker)
# ------------------------------------------------------------

def engineer_features(df):
    df = df.sort_values("date").copy()

    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # Moving averages
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["dist_ma_5"] = (df["close"] - df["ma_5"]) / df["ma_5"]
    df["dist_ma_20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

    # Volatility
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # Lagged returns
    df["ret_lag1"] = df["ret_1"].shift(1)
    df["ret_lag2"] = df["ret_1"].shift(2)

    # Forward return (target base)
    df["fwd_ret"] = df["close"].pct_change(HORIZON).shift(-HORIZON)

    return df


# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------

raw = download_price_data(TICKERS)
panel = to_panel(raw)

panel = (
    panel
    .groupby("ticker", group_keys=False)
    .apply(engineer_features)
)

# Cross-sectional target
panel["cs_rank"] = panel.groupby("date")["fwd_ret"].rank(pct=True)
panel["target"] = (panel["cs_rank"] > 0.6).astype(int)

panel = panel.dropna().reset_index(drop=True)

FEATURES = [
    "ret_1", "ret_5", "ret_10",
    "dist_ma_5", "dist_ma_20",
    "vol_5", "vol_20",
    "ret_lag1", "ret_lag2"
]

X = panel[FEATURES]
y = panel["target"]

# ------------------------------------------------------------
# Walk-forward split
# ------------------------------------------------------------

train_cut = panel["date"].quantile(0.6)
val_cut = panel["date"].quantile(0.8)

train_mask = panel["date"] < train_cut
val_mask = (panel["date"] >= train_cut) & (panel["date"] < val_cut)
test_mask = panel["date"] >= val_cut

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        C=0.05,
        solver="lbfgs"
    ))
])

model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

print("VALIDATION AUC:", roc_auc_score(y_val, val_prob))
print("TEST AUC:", roc_auc_score(y_test, test_prob))

# ------------------------------------------------------------
# Cross-sectional Trading Strategy
# ------------------------------------------------------------

df_test = panel[test_mask].copy()
df_test["prob"] = test_prob

# Cross-sectional signals
df_test["signal"] = 0
df_test.loc[
    df_test["prob"] >= df_test.groupby("date")["prob"].transform(lambda x: x.quantile(LONG_Q)),
    "signal"
] = 1

df_test.loc[
    df_test["prob"] <= df_test.groupby("date")["prob"].transform(lambda x: x.quantile(SHORT_Q)),
    "signal"
] = -1

# Shift signals (no look-ahead)
df_test["signal_shifted"] = df_test.groupby("ticker")["signal"].shift(1)

# Transaction costs
df_test["trade"] = df_test.groupby("ticker")["signal_shifted"].diff().abs()
df_test["cost"] = df_test["trade"] * TRANSACTION_COST

# Strategy returns
df_test["strategy_ret"] = (
    df_test["signal_shifted"] * df_test["ret_1"] - df_test["cost"]
)

portfolio_ret = df_test.groupby("date")["strategy_ret"].mean()
cum_ret = (1 + portfolio_ret).cumprod()

# ------------------------------------------------------------
# Performance Metrics
# ------------------------------------------------------------

def sharpe(x):
    return np.sqrt(252) * x.mean() / x.std()

def max_drawdown(cum):
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

print("\nBACKTEST RESULTS")
print(f"Sharpe Ratio: {sharpe(portfolio_ret):.2f}")
print(f"Max Drawdown: {max_drawdown(cum_ret):.2%}")
print(f"Total Return: {(cum_ret.iloc[-1]-1)*100:.2f}%")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(cum_ret, label="ML Strategy")
plt.title("Cross-Sectional Long/Short Strategy")
plt.ylabel("Growth of $1")
plt.xlabel("Date")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
