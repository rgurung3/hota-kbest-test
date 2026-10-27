"""Replay a corpus of cost matrices through a k-best assignment implementation.

The corpus is the .npz/.json pair written by dump_hota_matrices.py. For every
matrix this script computes a reference k-best list (textbook Murty on top of
scipy's Hungarian solver), runs your candidate, and compares them.

Candidate contract (--candidate package.module:function):

    function(cost: np.ndarray, k: int) -> [(total_cost, [(row, col), ...]), ...]

sorted by non-decreasing cost. Compare cost lists, not assignments: HOTA
matrices have many exact zeros, so ties are everywhere and equal-cost
assignments may legitimately come out in a different order.

Semantics (--semantics) must match what the candidate implements:
    full     every row of the smaller side is assigned (scipy / classic Murty)
    partial  any row may stay unassigned at cost 0 (fastmurty style)

Examples:
    python replay_kbest.py --selftest
    python replay_kbest.py corpus/mot17_liftt.npz --k 20 --limit 200
    python replay_kbest.py corpus/mot17_liftt.npz --candidate tyler_impl:kbest --k 20
"""
import argparse
import heapq
import importlib
import itertools
import json
import os
import re
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def _solve_node(base, inc, exc):
    """Cheapest assignment of every row of `base` (n <= m) with the pairs in
    `inc` forced and the pairs in `exc` forbidden; None if infeasible."""
    n, m = base.shape
    inc_rows = {r for r, _ in inc}
    inc_cols = {c for _, c in inc}
    rows = [r for r in range(n) if r not in inc_rows]
    cols = [c for c in range(m) if c not in inc_cols]
    fixed = sum(base[r, c] for r, c in inc)
    if not rows:
        return fixed, sorted(inc)
    sub = base[np.ix_(rows, cols)].copy()
    big = (np.abs(base).max() + 1.0) * (n + 1) * 10.0
    rpos = {r: i for i, r in enumerate(rows)}
    cpos = {c: j for j, c in enumerate(cols)}
    for r, c in exc:
        if r in rpos and c in cpos:
            sub[rpos[r], cpos[c]] = big
    ri, ci = linear_sum_assignment(sub)
    pairs = [(rows[i], cols[j]) for i, j in zip(ri, ci)]
    if any(p in exc for p in pairs):
        return None
    return fixed + sum(base[r, c] for r, c in pairs), sorted(list(inc) + pairs)


def murty_reference(cost, k, semantics="full"):
    """Textbook Murty: returns up to k (cost, pairs) in non-decreasing cost."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    transpose = False
    forbidden = frozenset()
    if semantics == "partial":
        # One private zero-cost "miss" column per row turns partial matchings
        # into full assignments, one-to-one.
        base = np.hstack([cost, np.zeros((n, n))])
        forbidden = frozenset((i, m + j) for i in range(n) for j in range(n) if i != j)
    elif n > m:
        base, transpose = cost.T, True
    else:
        base = cost

    first = _solve_node(base, frozenset(), forbidden)
    if first is None:
        return []
    heap = [(first[0], 0, first[1], frozenset(), forbidden)]
    counter = 1
    out = []
    while heap and len(out) < k:
        c, _, pairs, inc, exc = heapq.heappop(heap)
        if semantics == "partial":
            real = [(r, cc) for r, cc in pairs if cc < m]
        elif transpose:
            real = [(cc, r) for r, cc in pairs]
        else:
            real = pairs
        out.append((c, tuple(sorted(real))))
        free = [p for p in pairs if p not in inc]
        for i, p in enumerate(free):
            child = _solve_node(base, inc | frozenset(free[:i]), exc | {p})
            if child is not None:
                heapq.heappush(heap, (child[0], counter, child[1], inc | frozenset(free[:i]), exc | {p}))
                counter += 1
    return out


def brute_force(cost, k, semantics="full"):
    """Enumerate every assignment. Only for tiny matrices; ground truth for selftest."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    sols = []
    if semantics == "full":
        if n <= m:
            for cols in itertools.permutations(range(m), n):
                pairs = tuple(zip(range(n), cols))
                sols.append((sum(cost[r, c] for r, c in pairs), pairs))
        else:
            for rows in itertools.permutations(range(n), m):
                pairs = tuple(sorted(zip(rows, range(m))))
                sols.append((sum(cost[r, c] for r, c in pairs), pairs))
    else:
        def rec(r, used, pairs, total):
            if r == n:
                sols.append((total, tuple(pairs)))
                return
            rec(r + 1, used, pairs, total)
            for c in range(m):
                if c not in used:
                    rec(r + 1, used | {c}, pairs + [(r, c)], total + cost[r, c])
        rec(0, frozenset(), [], 0.0)
    sols.sort(key=lambda s: s[0])
    return sols[:k]


def check(cost, semantics, expected_costs, result):
    """Return a list of problems with `result` (empty list means it passes)."""
    n, m = cost.shape
    tol = 1e-9 * max(1.0, float(np.abs(cost).max())) * max(n, m)
    problems = []
    if len(result) != len(expected_costs):
        problems.append("returned %d solutions, expected %d" % (len(result), len(expected_costs)))
    seen = set()
    for i, (claimed, pairs) in enumerate(result[:len(expected_costs)]):
        pairs = tuple(sorted((int(r), int(c)) for r, c in pairs))
        rows = [r for r, _ in pairs]
        cols = [c for _, c in pairs]
        if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
            problems.append("solution %d reuses a row or column" % i)
        elif any(not (0 <= r < n and 0 <= c < m) for r, c in pairs):
            problems.append("solution %d has an out-of-range index" % i)
        elif semantics == "full" and len(pairs) != min(n, m):
            problems.append("solution %d assigns %d pairs, expected %d" % (i, len(pairs), min(n, m)))
        else:
            true_cost = sum(cost[r, c] for r, c in pairs)
            if abs(true_cost - claimed) > tol:
                problems.append("solution %d claims cost %.12g but its pairs sum to %.12g" % (i, claimed, true_cost))
        if pairs in seen:
            problems.append("solution %d duplicates an earlier assignment" % i)
        seen.add(pairs)
        if abs(claimed - expected_costs[i]) > tol:
            problems.append("solution %d cost %.12g, reference %.12g" % (i, claimed, expected_costs[i]))
    return problems


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_corpus(path):
    data = np.load(path)
    meta_path = os.path.splitext(path)[0] + ".json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = {r["key"]: r for r in json.load(f)}
    for key in sorted(data.files, key=natural_key):
        yield key, data[key], meta.get(key, {})


def selftest(trials=300, seed=0):
    rng = np.random.default_rng(seed)
    bad = 0
    for semantics in ("full", "partial"):
        for t in range(trials):
            n, m = rng.integers(1, 5), rng.integers(1, 5)
            # small integers on purpose: lots of ties, like HOTA matrices
            cost = rng.integers(-3, 1, size=(n, m)).astype(float)
            k = int(rng.integers(1, 40))
            ref = murty_reference(cost, k, semantics)
            truth = brute_force(cost, k, semantics)
            problems = check(cost, semantics, [c for c, _ in truth], ref)
            if problems:
                bad += 1
                print("MISMATCH", semantics, cost.tolist(), "k=%d" % k, problems[:2])
    print("selftest: %d mismatches in %d trials" % (bad, 2 * trials))
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus", nargs="?", help=".npz written by dump_hota_matrices.py")
    ap.add_argument("--candidate", help="package.module:function")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--semantics", choices=["full", "partial"], default="full")
    ap.add_argument("--limit", type=int, help="only the first N matrices")
    ap.add_argument("--min-dim", type=int, default=1, help="skip matrices whose smaller side is below this")
    ap.add_argument("--max-dim", type=int, help="skip matrices whose larger side exceeds this")
    ap.add_argument("--selftest", action="store_true", help="validate the reference against brute force and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.corpus:
        ap.error("corpus is required unless --selftest is given")

    candidate = None
    if args.candidate:
        sys.path.insert(0, os.getcwd())
        module_name, func_name = args.candidate.split(":")
        candidate = getattr(importlib.import_module(module_name), func_name)

    ran = failed = 0
    t_ref = t_cand = 0.0
    for key, cost, meta in load_corpus(args.corpus):
        n, m = cost.shape
        if min(n, m) < args.min_dim or (args.max_dim and max(n, m) > args.max_dim):
            continue
        if args.limit and ran >= args.limit:
            break
        t0 = time.perf_counter()
        ref = murty_reference(cost, args.k, args.semantics)
        t_ref += time.perf_counter() - t0
        ran += 1
        if candidate is None:
            continue
        t0 = time.perf_counter()
        result = candidate(cost, args.k)
        t_cand += time.perf_counter() - t0
        problems = check(cost, args.semantics, [c for c, _ in ref], result)
        if problems:
            failed += 1
            if failed <= 10:
                where = "%s frame %s" % (meta.get("seq", "?"), meta.get("frame", "?"))
                print("FAIL %s (%s, %dx%d): %s" % (key, where, n, m, "; ".join(problems[:3])))

    print("matrices run: %d, reference time %.2fs" % (ran, t_ref))
    if candidate is not None:
        print("candidate failures: %d, candidate time %.2fs" % (failed, t_cand))
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
