/*
 * orderbook.h — Limit Order Book Simulator
 *
 * Models a financial exchange order book with bids and asks.
 * Each price level holds a FIFO queue of orders (Queue ADT — S6).
 * Price levels are stored in a BST for O(log n) lookup (S9).
 * The order_book type is opaque (S6/S8).
 *
 * CS 136 concepts used:
 *   S6  — ADT design (Queue for order queue, Dictionary for price BST)
 *         opaque structs, .h/.c module split, static helpers
 *   S8  — malloc/free for every order and BST node
 *   S9  — BST indexed by price, linked-list Queue at each price level
 *   S10 — generic ADT design; choosing the right data structure
 */

#ifndef ORDERBOOK_H
#define ORDERBOOK_H

/* Order side */
typedef enum { SIDE_BID = 0, SIDE_ASK = 1 } order_side_t;

/* Opaque order book type (S6/S8) */
struct order_book;

/*
 * ob_create — create an empty order book.
 * effects: allocates heap memory [caller must call ob_destroy]
 */
struct order_book *ob_create(void);

/* Free all heap memory. No-op on NULL. */
void ob_destroy(struct order_book *ob);

/*
 * ob_add_order — submit a limit order to the book.
 *   order_id : unique identifier (positive integer)
 *   side     : SIDE_BID (buy) or SIDE_ASK (sell)
 *   price    : limit price in cents (integer to avoid float rounding)
 *   quantity : number of units
 *
 * Returns 1 on success, 0 on failure.
 * Orders at the same price are queued FIFO (Queue ADT — S6).
 * If the order crosses the spread, it is matched immediately.
 */
int ob_add_order(struct order_book *ob,
                 int order_id,
                 order_side_t side,
                 int price,
                 int quantity);

/*
 * ob_cancel_order — remove an order from the book by id.
 * Returns 1 if found and cancelled, 0 if not found.
 */
int ob_cancel_order(struct order_book *ob, int order_id);

/* Best bid price (-1 if no bids). */
int ob_best_bid(const struct order_book *ob);

/* Best ask price (-1 if no asks). */
int ob_best_ask(const struct order_book *ob);

/* Spread = best ask - best bid. Returns -1 if book is one-sided. */
int ob_spread(const struct order_book *ob);

/* Total number of resting orders on the given side. */
int ob_depth(const struct order_book *ob, order_side_t side);

/*
 * ob_print — print the order book to stdout (top N levels per side).
 */
void ob_print(const struct order_book *ob, int levels);

/* Total fills executed since creation. */
int ob_total_fills(const struct order_book *ob);

#endif /* ORDERBOOK_H */
