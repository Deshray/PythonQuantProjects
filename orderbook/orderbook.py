"""
orderbook.py — Limit Order Book: core matching engine

Implements a full price-time priority order book with:
  - Bid/ask sides using SortedDict (BST-equivalent O(log n) ops)
  - FIFO queue at each price level (deque)
  - Aggressive order matching with partial fills
  - Order cancellation by id
  - Trade history and analytics: VWAP, spread history, order flow imbalance
  - Market depth snapshot
"""

from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from sortedcontainers import SortedDict
import time


class Side(Enum):
    BID = "BID"
    ASK = "ASK"


@dataclass
class Order:
    order_id:  int
    side:      Side
    price:     float     # limit price
    quantity:  int       # remaining quantity
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trade:
    aggressor_id:  int
    resting_id:    int
    price:         float
    quantity:      int
    timestamp:     float = field(default_factory=time.time)


class LimitOrderBook:
    """
    Full limit order book with price-time priority matching.

    Internal structure:
      bids: SortedDict  price → deque[Order]  (highest price = best bid)
      asks: SortedDict  price → deque[Order]  (lowest price  = best ask)

    SortedDict is a BST-backed dict giving O(log n) insert/lookup/delete.
    deque gives O(1) append (enqueue) and popleft (dequeue) — FIFO.
    """

    def __init__(self, name: str = "Book"):
        self.name = name

        # BST-backed: bids sorted ascending so bids.keys()[-1] = best bid
        self._bids: SortedDict = SortedDict()
        # BST-backed: asks sorted ascending so asks.keys()[0]  = best ask
        self._asks: SortedDict = SortedDict()

        # Order id → (side, price) for O(1) cancellation lookup
        self._order_index: dict[int, tuple[Side, float]] = {}

        # Analytics
        self._trades:        list[Trade] = []
        self._spread_history: list[tuple[float, float]] = []  # (timestamp, spread)
        self._order_count_bid = 0
        self._order_count_ask = 0

    # ── Submission ────────────────────────────────────────────────────────────

    def add_order(self, order_id: int, side: Side,
                   price: float, quantity: int) -> list[Trade]:
        """
        Submit a limit order. Matches immediately if crossing; rests otherwise.
        Returns a list of Trade objects executed (may be empty).
        """
        if quantity <= 0 or price <= 0:
            raise ValueError("Quantity and price must be positive")
        if order_id in self._order_index:
            raise ValueError(f"Duplicate order_id {order_id}")

        order  = Order(order_id, side, price, quantity)
        trades = self._match(order)

        # Rest any unfilled quantity
        if order.quantity > 0:
            book = self._bids if side == Side.BID else self._asks
            if price not in book:
                book[price] = deque()
            book[price].append(order)
            self._order_index[order_id] = (side, price)
            if side == Side.BID:
                self._order_count_bid += 1
            else:
                self._order_count_ask += 1

        # Record spread snapshot
        s = self.spread()
        if s is not None:
            self._spread_history.append((time.time(), s))

        return trades

    # ── Cancellation ──────────────────────────────────────────────────────────

    def cancel_order(self, order_id: int) -> bool:
        """
        Remove a resting order by id. Returns True if found and removed.
        O(1) lookup via index, O(queue_len) removal from deque.
        """
        if order_id not in self._order_index:
            return False

        side, price = self._order_index.pop(order_id)
        book = self._bids if side == Side.BID else self._asks

        if price in book:
            queue = book[price]
            # Remove from deque (O(n) in deque length at that level)
            new_q = deque(o for o in queue if o.order_id != order_id)
            if new_q:
                book[price] = new_q
            else:
                del book[price]

        if side == Side.BID:
            self._order_count_bid -= 1
        else:
            self._order_count_ask -= 1

        return True

    # ── Matching Engine ───────────────────────────────────────────────────────

    def _match(self, aggressor: Order) -> list[Trade]:
        """
        Execute fills against the opposite side. Modifies aggressor.quantity
        in place and removes/reduces resting orders.
        """
        trades    = []
        opp_book  = self._asks if aggressor.side == Side.BID else self._bids

        while aggressor.quantity > 0 and opp_book:
            # Best opposing price: lowest ask for buyer, highest bid for seller
            best_price = (opp_book.keys()[0]  if aggressor.side == Side.BID
                          else opp_book.keys()[-1])

            # Check if crossing
            if aggressor.side == Side.BID and best_price > aggressor.price:
                break
            if aggressor.side == Side.ASK and best_price < aggressor.price:
                break

            queue = opp_book[best_price]

            while aggressor.quantity > 0 and queue:
                resting  = queue[0]
                fill_qty = min(aggressor.quantity, resting.quantity)

                trade = Trade(
                    aggressor_id = aggressor.order_id,
                    resting_id   = resting.order_id,
                    price        = best_price,
                    quantity     = fill_qty,
                )
                trades.append(trade)
                self._trades.append(trade)

                aggressor.quantity -= fill_qty
                resting.quantity   -= fill_qty

                if resting.quantity == 0:
                    queue.popleft()
                    self._order_index.pop(resting.order_id, None)
                    if aggressor.side == Side.BID:
                        self._order_count_ask -= 1
                    else:
                        self._order_count_bid -= 1

            if not queue:
                del opp_book[best_price]

        return trades

    # ── Market Data ───────────────────────────────────────────────────────────

    def best_bid(self) -> float | None:
        return self._bids.keys()[-1] if self._bids else None

    def best_ask(self) -> float | None:
        return self._asks.keys()[0] if self._asks else None

    def spread(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        return round(ba - bb, 6) if (bb and ba) else None

    def mid_price(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        return round((bb + ba) / 2, 6) if (bb and ba) else None

    def total_bid_qty(self) -> int:
        return sum(sum(o.quantity for o in q) for q in self._bids.values())

    def total_ask_qty(self) -> int:
        return sum(sum(o.quantity for o in q) for q in self._asks.values())

    def order_flow_imbalance(self) -> float:
        """
        OFI = (bid_qty - ask_qty) / (bid_qty + ask_qty).
        Range [-1, 1]: +1 = all buying pressure, -1 = all selling pressure.
        """
        bq = self.total_bid_qty()
        aq = self.total_ask_qty()
        total = bq + aq
        return round((bq - aq) / total, 4) if total > 0 else 0.0

    def depth_snapshot(self, levels: int = 5) -> dict:
        """
        Top N price levels per side with aggregated quantities.
        """
        bids = [
            {"price": p,
             "qty":   sum(o.quantity for o in q),
             "orders": len(q)}
            for p, q in reversed(self._bids.items())
        ][:levels]

        asks = [
            {"price": p,
             "qty":   sum(o.quantity for o in q),
             "orders": len(q)}
            for p, q in self._asks.items()
        ][:levels]

        return {"bids": bids, "asks": asks,
                "best_bid": self.best_bid(),
                "best_ask": self.best_ask(),
                "spread":   self.spread(),
                "mid":      self.mid_price()}

    # ── Analytics ─────────────────────────────────────────────────────────────

    def vwap(self) -> float | None:
        """Volume-Weighted Average Price across all executed trades."""
        if not self._trades:
            return None
        total_vol = sum(t.quantity for t in self._trades)
        if total_vol == 0:
            return None
        return round(
            sum(t.price * t.quantity for t in self._trades) / total_vol, 4
        )

    def trade_summary(self) -> dict:
        if not self._trades:
            return {"n_trades": 0}
        prices = [t.price    for t in self._trades]
        qtys   = [t.quantity for t in self._trades]
        return {
            "n_trades":    len(self._trades),
            "total_volume": sum(qtys),
            "vwap":         self.vwap(),
            "min_price":    min(prices),
            "max_price":    max(prices),
            "avg_price":    round(sum(p * q for p, q in zip(prices, qtys))
                                  / sum(qtys), 4),
        }

    def spread_history_df(self):
        """Return spread history as a DataFrame for plotting."""
        import pandas as pd
        if not self._spread_history:
            return pd.DataFrame(columns=["timestamp", "spread"])
        return pd.DataFrame(self._spread_history,
                             columns=["timestamp", "spread"])

    def trades_df(self):
        """All executed trades as a DataFrame."""
        import pandas as pd
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "aggressor_id": t.aggressor_id,
            "resting_id":   t.resting_id,
            "price":        t.price,
            "quantity":     t.quantity,
        } for t in self._trades])

    def reset(self):
        """Clear the book and all analytics."""
        self._bids.clear()
        self._asks.clear()
        self._order_index.clear()
        self._trades.clear()
        self._spread_history.clear()
        self._order_count_bid = 0
        self._order_count_ask = 0
