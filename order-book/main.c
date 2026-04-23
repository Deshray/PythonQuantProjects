/*
 * main.c — Order Book Simulator: test suite + trading scenario demo
 */

#include "orderbook.h"
#include <stdio.h>
#include <stdlib.h>

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define CHECK(expr, msg) \
    do { \
        tests_run++; \
        if (expr) { tests_passed++; printf("  [PASS] %s\n", msg); } \
        else      { tests_failed++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
    } while (0)

static void run_tests(void)
{
    printf("\n--- 1. Basic Bid/Ask ---\n");
    struct order_book *ob = ob_create();
    CHECK(ob != NULL, "ob_create returns non-NULL");
    CHECK(ob_best_bid(ob) == -1, "empty book: best bid is -1");
    CHECK(ob_best_ask(ob) == -1, "empty book: best ask is -1");

    ob_add_order(ob, 1, SIDE_BID, 100, 10);
    ob_add_order(ob, 2, SIDE_BID, 99,  5);
    ob_add_order(ob, 3, SIDE_ASK, 102, 8);
    ob_add_order(ob, 4, SIDE_ASK, 103, 3);

    CHECK(ob_best_bid(ob) == 100, "best bid = 100");
    CHECK(ob_best_ask(ob) == 102, "best ask = 102");
    CHECK(ob_spread(ob)   == 2,   "spread = 2");
    CHECK(ob_depth(ob, SIDE_BID) == 2, "bid depth = 2 orders");
    CHECK(ob_depth(ob, SIDE_ASK) == 2, "ask depth = 2 orders");

    ob_print(ob, 5);

    printf("--- 2. FIFO Matching (Queue ADT) ---\n");
    /* Two bids at 100, then a crossing ask should fill the oldest first */
    struct order_book *ob2 = ob_create();
    ob_add_order(ob2, 10, SIDE_BID, 100, 5);   /* oldest at 100 */
    ob_add_order(ob2, 11, SIDE_BID, 100, 3);   /* newer at 100  */
    printf("  Adding crossing ASK @ 100 qty=6 (should fill bid 10 fully, bid 11 partial):\n");
    ob_add_order(ob2, 12, SIDE_ASK, 100, 6);
    CHECK(ob_total_fills(ob2) == 2, "2 fills executed (FIFO: 10 then 11)");
    CHECK(ob_depth(ob2, SIDE_BID) == 1, "1 bid order remains");
    ob_print(ob2, 5);
    ob_destroy(ob2);

    printf("--- 3. Cancellation ---\n");
    CHECK(ob_cancel_order(ob, 2) == 1, "cancel order 2 returns 1 (found)");
    CHECK(ob_cancel_order(ob, 99) == 0, "cancel nonexistent order returns 0");
    CHECK(ob_best_bid(ob) == 100, "best bid still 100 after cancelling 99-bid");
    CHECK(ob_depth(ob, SIDE_BID) == 1, "bid depth now 1");

    printf("--- 4. Full Fill ---\n");
    struct order_book *ob3 = ob_create();
    ob_add_order(ob3, 20, SIDE_BID, 200, 10);
    printf("  Adding exact crossing ASK @ 200 qty=10:\n");
    ob_add_order(ob3, 21, SIDE_ASK, 200, 10);
    CHECK(ob_depth(ob3, SIDE_BID) == 0, "all bids consumed");
    CHECK(ob_depth(ob3, SIDE_ASK) == 0, "ask fully matched, no resting ask");
    CHECK(ob_total_fills(ob3) == 1, "1 fill executed");
    ob_destroy(ob3);

    printf("--- 5. Null Safety ---\n");
    ob_destroy(NULL);
    CHECK(1, "ob_destroy(NULL) does not crash");
    CHECK(ob_best_bid(NULL) == -1, "ob_best_bid(NULL) = -1");
    CHECK(ob_spread(NULL) == -1, "ob_spread(NULL) = -1");

    ob_destroy(ob);

    printf("--- 6. Trading Scenario ---\n");
    struct order_book *market = ob_create();
    printf("  Building a realistic market...\n");
    ob_add_order(market, 101, SIDE_ASK, 15050, 100);
    ob_add_order(market, 102, SIDE_ASK, 15025, 200);
    ob_add_order(market, 103, SIDE_ASK, 15010, 50);
    ob_add_order(market, 104, SIDE_BID, 14990, 150);
    ob_add_order(market, 105, SIDE_BID, 14975, 300);
    ob_add_order(market, 106, SIDE_BID, 14960, 100);

    ob_print(market, 5);
    CHECK(ob_spread(market) == 20, "spread = 20 cents");

    printf("  Aggressive BID crossing the ask:\n");
    ob_add_order(market, 200, SIDE_BID, 15010, 50);
    CHECK(ob_total_fills(market) >= 1, "fill executed on aggressive bid");
    ob_print(market, 5);

    ob_destroy(market);
}

int main(void)
{
    printf("========================================\n");
    printf("   Limit Order Book — Test Suite\n");
    printf("========================================\n");

    run_tests();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf("  (%d FAILED)", tests_failed);
    printf("\n========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
