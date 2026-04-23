/*
 * orderbook.c — Limit Order Book Simulator Implementation
 *
 * Architecture:
 *   - Two BSTs (one for bids, one for asks), each a BST of price_level nodes
 *     sorted by price (S9 BST: left < node < right).
 *   - At each price_level, a linked-list Queue of orders (S6 Queue ADT, S9).
 *   - Matching: when a new order arrives, check if it crosses the best
 *     opposite price; if so, execute fills until the order is filled or
 *     no more crossing levels remain.
 *
 * CS 136 concepts demonstrated:
 *   S6  : Queue ADT (FIFO order queue at each price level)
 *   S8  : every node is malloc'd; every cancel/fill free's the order
 *   S9  : BST for price levels; linked-list Queue for order queue
 *   S10 : choosing the right data structure (BST for O(log n) price lookup)
 */

#include "orderbook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Order node (Queue element — S6/S9) ───────────────────────────────────── */

struct order_node {
    int order_id;
    int price;
    int quantity;
    order_side_t side;
    struct order_node *next;   /* next in FIFO queue */
};

/* ── Price level (BST node + Queue wrapper — S6/S9) ──────────────────────── */

struct price_level {
    int price;

    /* Queue (linked list) of orders at this price — FIFO (S6 Queue ADT) */
    struct order_node *front;   /* remove from front (oldest) */
    struct order_node *back;    /* add to back (newest)       */
    int order_count;

    /* BST links (S9) */
    struct price_level *left;
    struct price_level *right;
};

/* ── Opaque order book struct (S6/S8) ─────────────────────────────────────── */

struct order_book {
    struct price_level *bids;    /* BST root — buy orders  */
    struct price_level *asks;    /* BST root — sell orders */
    int total_fills;
    int total_volume_filled;
};

/* ── Static helpers — module scope (S6) ───────────────────────────────────── */

/*
 * pl_create — allocate a new price level node.
 * effects: allocates heap memory
 */
static struct price_level *pl_create(int price)
{
    struct price_level *pl = malloc(sizeof(struct price_level));
    if (!pl) return NULL;
    pl->price       = price;
    pl->front       = NULL;
    pl->back        = NULL;
    pl->order_count = 0;
    pl->left        = NULL;
    pl->right       = NULL;
    return pl;
}

/*
 * pl_enqueue — add an order to the back of the FIFO queue (S6 Queue ADT).
 */
static void pl_enqueue(struct price_level *pl, struct order_node *order)
{
    order->next = NULL;
    if (!pl->back) {
        pl->front = order;
        pl->back  = order;
    } else {
        pl->back->next = order;
        pl->back       = order;
    }
    pl->order_count++;
}

/*
 * pl_dequeue_front — remove and return the front order (S6 Queue ADT).
 * Returns NULL if the queue is empty.
 */
static struct order_node *pl_dequeue_front(struct price_level *pl)
{
    if (!pl->front) return NULL;
    struct order_node *order = pl->front;
    pl->front = order->next;
    if (!pl->front) pl->back = NULL;
    order->next = NULL;
    pl->order_count--;
    return order;
}

/*
 * bst_find — find a price level in the BST by price (S9).
 * Returns the node or NULL.
 * Time: O(h) where h = BST height.
 */
static struct price_level *bst_find(struct price_level *root, int price)
{
    struct price_level *cur = root;
    while (cur) {
        if      (price < cur->price) cur = cur->left;
        else if (price > cur->price) cur = cur->right;
        else                         return cur;
    }
    return NULL;
}

/*
 * bst_insert — insert a new price level into the BST (S9).
 * Returns the new root.
 * Time: O(h).
 */
static struct price_level *bst_insert(struct price_level *root,
                                       struct price_level *new_pl)
{
    if (!root) return new_pl;
    if (new_pl->price < root->price)
        root->left  = bst_insert(root->left,  new_pl);
    else if (new_pl->price > root->price)
        root->right = bst_insert(root->right, new_pl);
    /* duplicate price → should never happen (we check first with bst_find) */
    return root;
}

/*
 * bst_min — leftmost node (smallest price) in BST (S9).
 * Used to find the best ask.
 */
static struct price_level *bst_min(struct price_level *root)
{
    if (!root) return NULL;
    while (root->left) root = root->left;
    return root;
}

/*
 * bst_max — rightmost node (largest price) in BST (S9).
 * Used to find the best bid.
 */
static struct price_level *bst_max(struct price_level *root)
{
    if (!root) return NULL;
    while (root->right) root = root->right;
    return root;
}

/*
 * bst_remove_min — remove the smallest node (S9 BST removal).
 * Returns the new root.
 */
static struct price_level *bst_remove_min(struct price_level *root,
                                           struct price_level **removed)
{
    if (!root) { *removed = NULL; return NULL; }
    if (!root->left) {
        *removed = root;
        return root->right;
    }
    root->left = bst_remove_min(root->left, removed);
    return root;
}

/*
 * bst_remove_max — remove the largest node (S9 BST removal).
 * Returns the new root.
 */
static struct price_level *bst_remove_max(struct price_level *root,
                                           struct price_level **removed)
{
    if (!root) { *removed = NULL; return NULL; }
    if (!root->right) {
        *removed = root;
        return root->left;
    }
    root->right = bst_remove_max(root->right, removed);
    return root;
}

/*
 * bst_destroy — recursively free all price levels and their order queues (S8/S9).
 * One free per malloc — every order_node and every price_level.
 */
static void bst_destroy(struct price_level *root)
{
    if (!root) return;
    bst_destroy(root->left);
    bst_destroy(root->right);

    /* Free all order nodes in the queue at this level */
    struct order_node *cur = root->front;
    while (cur) {
        struct order_node *next = cur->next;
        free(cur);
        cur = next;
    }
    free(root);
}

/*
 * bst_depth — count total orders in the BST (tree traversal S9).
 */
static int bst_depth(const struct price_level *root)
{
    if (!root) return 0;
    return root->order_count + bst_depth(root->left) + bst_depth(root->right);
}

/*
 * ob_get_or_create_level — find or create the price level in the given BST.
 * Updates *root if a new node is inserted.
 */
static struct price_level *ob_get_or_create_level(struct price_level **root,
                                                    int price)
{
    struct price_level *pl = bst_find(*root, price);
    if (!pl) {
        pl = pl_create(price);
        if (!pl) return NULL;
        *root = bst_insert(*root, pl);
    }
    return pl;
}

/*
 * try_match — attempt to match the incoming order against the opposite side.
 * Executes fills until the order is exhausted or no crossing levels remain.
 * Returns the remaining unfilled quantity.
 */
static int try_match(struct order_book *ob,
                     order_side_t incoming_side,
                     int incoming_price,
                     int incoming_qty)
{
    int remaining = incoming_qty;

    while (remaining > 0) {
        struct price_level *best_opp = NULL;

        if (incoming_side == SIDE_BID) {
            /* Buyer: match against cheapest ask */
            best_opp = bst_min(ob->asks);
            if (!best_opp || best_opp->price > incoming_price) break;
        } else {
            /* Seller: match against highest bid */
            best_opp = bst_max(ob->bids);
            if (!best_opp || best_opp->price < incoming_price) break;
        }

        /* Execute fills at this price level (FIFO — S6 Queue) */
        while (remaining > 0 && best_opp->front) {
            struct order_node *resting = best_opp->front;
            int fill_qty = (resting->quantity < remaining)
                           ? resting->quantity : remaining;

            printf("  FILL: order %d (%s @ %d) fills %d units @ %d\n",
                   resting->order_id,
                   (resting->side == SIDE_BID) ? "BID" : "ASK",
                   resting->price, fill_qty, best_opp->price);

            resting->quantity -= fill_qty;
            remaining         -= fill_qty;
            ob->total_fills++;
            ob->total_volume_filled += fill_qty;

            if (resting->quantity == 0) {
                struct order_node *done = pl_dequeue_front(best_opp);
                free(done);   /* one free for one malloc — S8 */
            }
        }

        /* If this price level is now empty, remove it from BST (S9) */
        if (!best_opp->front) {
            struct price_level *removed = NULL;
            if (incoming_side == SIDE_BID)
                ob->asks = bst_remove_min(ob->asks, &removed);
            else
                ob->bids = bst_remove_max(ob->bids, &removed);
            if (removed) free(removed);   /* free the BST node itself — S8 */
        }
    }

    return remaining;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

struct order_book *ob_create(void)
{
    struct order_book *ob = malloc(sizeof(struct order_book));
    if (!ob) return NULL;
    ob->bids               = NULL;
    ob->asks               = NULL;
    ob->total_fills        = 0;
    ob->total_volume_filled = 0;
    return ob;
}

void ob_destroy(struct order_book *ob)
{
    if (!ob) return;
    bst_destroy(ob->bids);
    bst_destroy(ob->asks);
    free(ob);
}

int ob_add_order(struct order_book *ob,
                 int order_id,
                 order_side_t side,
                 int price,
                 int quantity)
{
    if (!ob || quantity <= 0 || price <= 0) return 0;

    /* Attempt matching first */
    int remaining = try_match(ob, side, price, quantity);
    if (remaining == 0) return 1;   /* fully filled — no resting order */

    /* Rest the unfilled portion in the book */
    struct price_level **tree = (side == SIDE_BID) ? &ob->bids : &ob->asks;
    struct price_level *pl = ob_get_or_create_level(tree, price);
    if (!pl) return 0;

    struct order_node *order = malloc(sizeof(struct order_node));
    if (!order) return 0;

    order->order_id = order_id;
    order->price    = price;
    order->quantity = remaining;
    order->side     = side;
    order->next     = NULL;

    pl_enqueue(pl, order);   /* FIFO enqueue — S6 Queue ADT */
    return 1;
}

int ob_cancel_order(struct order_book *ob, int order_id)
{
    if (!ob) return 0;

    /* Search both sides' BSTs — O(n) in worst case */
    struct price_level *sides[2] = { ob->bids, ob->asks };

    for (int s = 0; s < 2; s++) {
        /* BFS/DFS traversal of the BST to find the order */
        /* Use a simple iterative in-order with a helper stack */
        struct price_level *stack[256];
        int top = 0;
        struct price_level *cur = sides[s];

        while (cur || top > 0) {
            while (cur) { stack[top++] = cur; cur = cur->left; }
            cur = stack[--top];

            /* Search the queue at this level */
            struct order_node *prev = NULL;
            struct order_node *node = cur->front;
            while (node) {
                if (node->order_id == order_id) {
                    /* Remove from linked list */
                    if (prev) prev->next = node->next;
                    else      cur->front = node->next;
                    if (cur->back == node) cur->back = prev;
                    cur->order_count--;
                    free(node);   /* one free for one malloc — S8 */
                    return 1;
                }
                prev = node;
                node = node->next;
            }
            cur = cur->right;
        }
    }
    return 0;   /* not found */
}

int ob_best_bid(const struct order_book *ob)
{
    if (!ob) return -1;
    struct price_level *pl = bst_max(ob->bids);
    return pl ? pl->price : -1;
}

int ob_best_ask(const struct order_book *ob)
{
    if (!ob) return -1;
    struct price_level *pl = bst_min(ob->asks);
    return pl ? pl->price : -1;
}

int ob_spread(const struct order_book *ob)
{
    int bid = ob_best_bid(ob);
    int ask = ob_best_ask(ob);
    if (bid < 0 || ask < 0) return -1;
    return ask - bid;
}

int ob_depth(const struct order_book *ob, order_side_t side)
{
    if (!ob) return 0;
    return bst_depth(side == SIDE_BID ? ob->bids : ob->asks);
}

int ob_total_fills(const struct order_book *ob)
{
    return ob ? ob->total_fills : 0;
}

/*
 * print_side — print up to `levels` price levels from one side (S9 in-order).
 * Bids: descending (highest first).  Asks: ascending (lowest first).
 */
static void print_levels_desc(struct price_level *root, int *remaining)
{
    if (!root || *remaining <= 0) return;
    print_levels_desc(root->right, remaining);
    if (*remaining > 0) {
        printf("  BID  %6d  qty=%d\n",
               root->price, root->order_count);
        (*remaining)--;
    }
    print_levels_desc(root->left, remaining);
}

static void print_levels_asc(struct price_level *root, int *remaining)
{
    if (!root || *remaining <= 0) return;
    print_levels_asc(root->left, remaining);
    if (*remaining > 0) {
        printf("  ASK  %6d  qty=%d\n",
               root->price, root->order_count);
        (*remaining)--;
    }
    print_levels_asc(root->right, remaining);
}

void ob_print(const struct order_book *ob, int levels)
{
    if (!ob) return;
    printf("\n=== Order Book (%d levels) ==================\n", levels);
    printf("  Side   Price  Qty\n");
    printf("  ----   -----  ---\n");

    int r_ask = levels;
    print_levels_asc(ob->asks, &r_ask);
    printf("  ---------- spread: %d ----------\n", ob_spread(ob));
    int r_bid = levels;
    print_levels_desc(ob->bids, &r_bid);

    printf("  Total fills: %d  |  Volume: %d\n",
           ob->total_fills, ob->total_volume_filled);
    printf("============================================\n\n");
}
