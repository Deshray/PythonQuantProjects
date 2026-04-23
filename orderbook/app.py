"""
app.py — Limit Order Book Simulator: Streamlit dashboard

Two modes:
  1. Manual — submit and cancel orders interactively, watch fills happen
  2. Simulation — generate a random market and replay it in real time

Displays: live order book depth chart, VWAP, spread history,
order flow imbalance, and trade log.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random

from orderbook import LimitOrderBook, Side

st.set_page_config(
    page_title="Order Book Simulator",
    page_icon="📈",
    layout="wide",
)
st.title("📈 Limit Order Book Simulator")
st.caption("Price-time priority matching engine with VWAP, spread tracking, "
           "order flow imbalance, and market depth visualisation.")

# ── Session state ─────────────────────────────────────────────────────────────

if "ob" not in st.session_state:
    st.session_state.ob = LimitOrderBook("DEMO")
if "next_id" not in st.session_state:
    st.session_state.next_id = 1
if "log" not in st.session_state:
    st.session_state.log = []

ob: LimitOrderBook = st.session_state.ob


def get_id() -> int:
    oid = st.session_state.next_id
    st.session_state.next_id += 1
    return oid


def add_log(msg: str):
    st.session_state.log.insert(0, f"[{time.strftime('%H:%M:%S')}]  {msg}")
    if len(st.session_state.log) > 80:
        st.session_state.log = st.session_state.log[:80]


# ── Sidebar: mode & controls ──────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️  Controls")
    mode = st.radio("Mode", ["Manual", "Simulation"])

    st.divider()

    if mode == "Manual":
        st.subheader("Submit Order")
        o_side  = st.selectbox("Side", ["BID", "ASK"])
        o_price = st.number_input("Price", value=100.0, step=0.5, min_value=0.01)
        o_qty   = st.number_input("Quantity", value=10, step=1, min_value=1)
        if st.button("📤 Submit", use_container_width=True):
            side = Side.BID if o_side == "BID" else Side.ASK
            oid  = get_id()
            try:
                trades = ob.add_order(oid, side, float(o_price), int(o_qty))
                if trades:
                    for t in trades:
                        add_log(f"FILL  {t.quantity} @ {t.price:.2f}  "
                                f"(agg={t.aggressor_id}, rest={t.resting_id})")
                else:
                    add_log(f"REST  #{oid} {o_side} {o_qty} @ {o_price:.2f}")
            except ValueError as e:
                st.error(str(e))

        st.subheader("Cancel Order")
        cancel_id = st.number_input("Order ID to cancel", value=1, step=1, min_value=1)
        if st.button("❌ Cancel", use_container_width=True):
            ok = ob.cancel_order(int(cancel_id))
            add_log(f"CANCEL #{cancel_id}  →  {'OK' if ok else 'NOT FOUND'}")

    else:  # Simulation mode
        st.subheader("Simulation Parameters")
        sim_n       = st.slider("Orders to generate", 10, 500, 100)
        mid         = st.number_input("Starting mid price", value=100.0, step=1.0)
        tick        = st.number_input("Tick size", value=0.5, step=0.1, min_value=0.01)
        spread_half = st.slider("Half-spread (ticks)", 1, 10, 2)
        agr_ratio   = st.slider("Aggressive order ratio", 0.0, 1.0, 0.3, 0.05)
        seed        = st.number_input("Random seed", value=42, step=1)

        if st.button("▶  Run Simulation", use_container_width=True):
            ob.reset()
            st.session_state.next_id = 1
            st.session_state.log     = []
            add_log("--- Simulation started ---")

            rng = random.Random(seed)
            cur_mid = mid
            n_fills = 0

            for i in range(sim_n):
                # Random walk on mid price
                cur_mid += rng.gauss(0, tick)
                cur_mid  = max(tick, round(cur_mid / tick) * tick)

                side    = Side.BID if rng.random() < 0.5 else Side.ASK
                qty     = rng.randint(1, 20)
                is_aggr = rng.random() < agr_ratio

                if is_aggr:
                    # Aggressive: cross the spread
                    price = (cur_mid + spread_half * tick
                             if side == Side.BID
                             else cur_mid - spread_half * tick)
                else:
                    # Passive: behind the mid
                    offset = rng.randint(1, 5) * tick
                    price  = (cur_mid - offset if side == Side.BID
                               else cur_mid + offset)

                price = max(tick, round(price / tick) * tick)
                oid   = get_id()
                try:
                    trades = ob.add_order(oid, side, price, qty)
                    n_fills += len(trades)
                except ValueError:
                    pass

            add_log(f"Simulation complete: {sim_n} orders, {n_fills} fills")

    st.divider()
    if st.button("🗑  Reset Book", use_container_width=True):
        ob.reset()
        st.session_state.next_id = 1
        st.session_state.log     = []
        add_log("Book reset.")

# ── Main dashboard ────────────────────────────────────────────────────────────

depth = ob.depth_snapshot(levels=8)
summary = ob.trade_summary()

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Best Bid",  f"{depth['best_bid']:.2f}" if depth['best_bid'] else "—")
m2.metric("Best Ask",  f"{depth['best_ask']:.2f}" if depth['best_ask'] else "—")
m3.metric("Spread",    f"{depth['spread']:.4f}"   if depth['spread']   else "—")
m4.metric("VWAP",      f"{summary['vwap']:.4f}"  if summary.get('vwap') else "—")
m5.metric("OFI",       f"{ob.order_flow_imbalance():+.3f}")

# ── Order Book Depth Chart ────────────────────────────────────────────────────

st.subheader("📗 Order Book Depth")

left_col, right_col = st.columns([3, 2])

with left_col:
    if depth["bids"] or depth["asks"]:
        # Cumulative depth chart
        bid_prices = [b["price"] for b in depth["bids"]]
        bid_qtys   = [b["qty"]   for b in depth["bids"]]
        ask_prices = [a["price"] for a in depth["asks"]]
        ask_qtys   = [a["qty"]   for a in depth["asks"]]

        # Accumulate from best outward
        cum_bid = list(np.cumsum(bid_qtys))
        cum_ask = list(np.cumsum(ask_qtys))

        fig_depth = go.Figure()
        fig_depth.add_trace(go.Bar(
            x=bid_prices, y=cum_bid, name="Bids",
            marker_color="rgba(0,180,100,0.7)",
            orientation="v",
        ))
        fig_depth.add_trace(go.Bar(
            x=ask_prices, y=cum_ask, name="Asks",
            marker_color="rgba(220,50,50,0.7)",
            orientation="v",
        ))
        if depth["mid"]:
            fig_depth.add_vline(x=depth["mid"], line_dash="dot",
                                 line_color="gold",
                                 annotation_text=f"Mid {depth['mid']:.2f}")
        fig_depth.update_layout(
            barmode="overlay", height=350,
            xaxis_title="Price", yaxis_title="Cumulative Qty",
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=30, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_depth, use_container_width=True)
    else:
        st.info("Book is empty — submit some orders.")

with right_col:
    st.markdown("**Bid levels**")
    if depth["bids"]:
        bid_df = pd.DataFrame(depth["bids"]).rename(
            columns={"price": "Price", "qty": "Qty", "orders": "Orders"})
        st.dataframe(bid_df.style.background_gradient(
            subset=["Qty"], cmap="Greens"),
            hide_index=True, use_container_width=True)
    st.markdown("**Ask levels**")
    if depth["asks"]:
        ask_df = pd.DataFrame(depth["asks"]).rename(
            columns={"price": "Price", "qty": "Qty", "orders": "Orders"})
        st.dataframe(ask_df.style.background_gradient(
            subset=["Qty"], cmap="Reds"),
            hide_index=True, use_container_width=True)

# ── Trade Analytics ───────────────────────────────────────────────────────────

if summary.get("n_trades", 0) > 0:
    st.subheader("📉 Trade Analytics")

    ta1, ta2, ta3, ta4 = st.columns(4)
    ta1.metric("Trades",        summary["n_trades"])
    ta2.metric("Total Volume",  summary["total_volume"])
    ta3.metric("VWAP",          f"{summary['vwap']:.4f}")
    ta4.metric("Price Range",   f"{summary['min_price']:.2f} – {summary['max_price']:.2f}")

    trades_df = ob.trades_df()
    if not trades_df.empty:
        fig_trades = go.Figure()
        fig_trades.add_trace(go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df["price"],
            mode="lines+markers",
            marker=dict(size=trades_df["quantity"],
                         sizemode="area", sizeref=max(trades_df["quantity"]) / 15,
                         color="#4C72B0"),
            line=dict(color="#4C72B0", width=1.5),
            name="Trade price",
        ))
        if summary.get("vwap"):
            fig_trades.add_hline(y=summary["vwap"], line_dash="dash",
                                  line_color="orange",
                                  annotation_text=f"VWAP {summary['vwap']:.2f}")
        fig_trades.update_layout(
            height=280, xaxis_title="Trade #", yaxis_title="Price",
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_trades, use_container_width=True)

# ── Spread History ────────────────────────────────────────────────────────────

    spread_df = ob.spread_history_df()
    if len(spread_df) > 1:
        fig_spread = px.line(spread_df, x="timestamp", y="spread",
                              title="Spread over time",
                              labels={"spread": "Spread", "timestamp": ""})
        fig_spread.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_spread, use_container_width=True)

# ── Event Log ─────────────────────────────────────────────────────────────────

st.subheader("📋 Event Log")
log_text = "\n".join(st.session_state.log[:30]) if st.session_state.log else "No events yet."
st.code(log_text, language=None)
