#!/usr/bin/env python3
"""
kelly.py — bet-sizing practice against a live bankroll.

Each hand deals you a gamble described by its outcomes. You say what fraction
of your bankroll to stake. The house tells you the log-optimal fraction, how
much growth you gave up, and then rolls the dice for real.

Run:      python3 kelly.py
Options:  python3 kelly.py --hands 20 --level 3 --seed 7 --terse

Levels
  1  binary, you lose the whole stake when you lose
  2  binary, but the losing branch is partial (this is where intuition dies)
  3  three or four discrete outcomes
  4  discrete outcomes plus a uniform range, sometimes with a total-loss branch
  0  mixed bag drawn from all of the above  (default)

Negative-edge hands appear at every level. The correct answer to those is zero.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Representing a gamble
# --------------------------------------------------------------------------
#
# Everything is stated as NET RETURN PER UNIT STAKED.
#   +1.0  means you win an amount equal to your stake (stake doubles)
#   -1.0  means the stake is gone
#   +0.2  means the stake grows 20%
#   -0.15 means the stake shrinks 15%
#
# A gamble is a list of discrete atoms (prob, net return) plus a list of
# uniform bands (prob, lo, hi) where the net return is uniform on [lo, hi].

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(96)


@dataclass
class Gamble:
    label: str
    atoms: list[tuple[float, float]] = field(default_factory=list)
    bands: list[tuple[float, float, float]] = field(default_factory=list)

    # ---- basic descriptive stats -----------------------------------------

    def total_prob(self) -> float:
        return sum(p for p, _ in self.atoms) + sum(p for p, _, _ in self.bands)

    def edge(self) -> float:
        """Expected net return per unit staked. The naive 'edge'."""
        e = sum(p * r for p, r in self.atoms)
        e += sum(p * (lo + hi) / 2 for p, lo, hi in self.bands)
        return e

    def variance(self) -> float:
        m = self.edge()
        v = sum(p * (r - m) ** 2 for p, r in self.atoms)
        for p, lo, hi in self.bands:
            # E[X^2] over the band = (hi^3 - lo^3) / (3(hi-lo))
            ex2 = (hi**3 - lo**3) / (3 * (hi - lo))
            v += p * (ex2 - 2 * m * (lo + hi) / 2 + m * m)
        return v

    def worst_loss(self) -> float:
        """Magnitude of the worst possible loss per unit staked."""
        worst = 0.0
        for _, r in self.atoms:
            worst = max(worst, -r)
        for _, lo, _ in self.bands:
            worst = max(worst, -lo)
        return worst

    def max_fraction(self) -> float:
        """Largest stake that cannot wipe you out. 1 / worst_loss."""
        w = self.worst_loss()
        return math.inf if w <= 0 else 1.0 / w

    # ---- the log-growth machinery ----------------------------------------

    def growth(self, f: float) -> float:
        """g(f) = E[log(1 + f * r)], the long-run growth rate per hand."""
        if f < 0:
            return -math.inf
        total = 0.0
        for p, r in self.atoms:
            x = 1 + f * r
            if x <= 1e-12:
                return -math.inf
            total += p * math.log(x)
        for p, lo, hi in self.bands:
            mid, half = (hi + lo) / 2, (hi - lo) / 2
            pts = mid + half * GL_NODES
            xs = 1 + f * pts
            if np.any(xs <= 1e-12):
                return -math.inf
            total += p * float(np.dot(GL_WEIGHTS, np.log(xs))) / 2
        return total

    def dgrowth(self, f: float) -> float:
        """g'(f). Decreasing in f, so a sign change is all we need."""
        total = 0.0
        for p, r in self.atoms:
            x = 1 + f * r
            if x <= 1e-12:
                return -math.inf
            total += p * r / x
        for p, lo, hi in self.bands:
            mid, half = (hi + lo) / 2, (hi - lo) / 2
            pts = mid + half * GL_NODES
            xs = 1 + f * pts
            if np.any(xs <= 1e-12):
                return -math.inf
            total += p * float(np.dot(GL_WEIGHTS, pts / xs)) / 2
        return total

    def kelly(self, leverage_cap: float = 1.0) -> float:
        """
        The growth-optimal stake, capped at `leverage_cap` (1.0 = no borrowing).
        Returns 0.0 when the gamble has non-positive edge.
        """
        if self.dgrowth(0.0) <= 0:
            return 0.0
        upper = min(self.max_fraction() * (1 - 1e-9), leverage_cap)
        if self.dgrowth(upper) >= 0:
            return upper  # wants more than it's allowed to have
        lo, hi = 0.0, upper
        for _ in range(200):
            mid = (lo + hi) / 2
            if self.dgrowth(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # ---- sampling ---------------------------------------------------------

    def draw(self, rng: random.Random) -> float:
        u = rng.random()
        acc = 0.0
        for p, r in self.atoms:
            acc += p
            if u <= acc:
                return r
        for p, lo, hi in self.bands:
            acc += p
            if u <= acc:
                return rng.uniform(lo, hi)
        return self.atoms[-1][1] if self.atoms else 0.0

    # ---- display ----------------------------------------------------------

    def describe(self) -> str:
        parts = []
        for p, r in self.atoms:
            if abs(r + 1) < 1e-9:
                what = "you lose the whole stake"
            elif r > 0:
                what = f"stake grows {r * 100:.0f}%"
            else:
                what = f"stake shrinks {-r * 100:.0f}%"
            parts.append(f"{p * 100:.0f}%: {what}")
        for p, lo, hi in self.bands:
            parts.append(
                f"{p * 100:.0f}%: stake changes by a uniform draw from "
                f"{lo * 100:+.0f}% to {hi * 100:+.0f}%"
            )
        return "\n".join("    " + s for s in parts)


# --------------------------------------------------------------------------
# Hand generation
# --------------------------------------------------------------------------


def _norm(atoms):
    s = sum(p for p, _ in atoms)
    return [(p / s, r) for p, r in atoms]


def gen_level_1(rng: random.Random) -> Gamble:
    """Binary, total loss on the downside. The textbook case."""
    b = rng.choice([0.5, 1.0, 1.0, 1.5, 2.0, 3.0])          # net odds if you win
    p = round(rng.uniform(0.25, 0.80), 2)
    return Gamble(f"binary, total loss (b={b:g})", atoms=[(p, b), (1 - p, -1.0)])


def gen_level_2(rng: random.Random) -> Gamble:
    """Binary with a partial loss. Kelly gets very large here."""
    up = round(rng.uniform(0.05, 0.60), 2)
    down = round(rng.uniform(0.05, min(0.60, up * rng.uniform(0.4, 1.6))), 2)
    p = round(rng.uniform(0.35, 0.70), 2)
    return Gamble("binary, partial loss", atoms=[(p, up), (1 - p, -down)])


def gen_level_3(rng: random.Random) -> Gamble:
    """Three or four discrete outcomes, sometimes with a wipeout branch."""
    n = rng.choice([3, 3, 4])
    atoms = []
    for _ in range(n - 1):
        r = round(rng.uniform(-0.6, 1.5), 2)
        atoms.append((rng.uniform(0.1, 0.5), r))
    tail = -1.0 if rng.random() < 0.45 else round(-rng.uniform(0.1, 0.7), 2)
    atoms.append((rng.uniform(0.1, 0.4), tail))
    rng.shuffle(atoms)
    return Gamble(f"{n} discrete outcomes", atoms=_norm(atoms))


def gen_level_4(rng: random.Random) -> Gamble:
    """Discrete atoms plus a uniform band."""
    atoms = [(rng.uniform(0.15, 0.35), round(rng.uniform(0.5, 2.0), 2))]
    if rng.random() < 0.6:
        atoms.append((rng.uniform(0.15, 0.35), -1.0))
    else:
        atoms.append((rng.uniform(0.15, 0.35), round(-rng.uniform(0.2, 0.8), 2)))
    lo = round(rng.uniform(-0.5, 0.1), 2)
    hi = round(lo + rng.uniform(0.4, 1.4), 2)
    band_p = max(0.15, 1 - sum(p for p, _ in atoms))
    raw = atoms + [(band_p, None)]
    s = sum(p for p, _ in raw)
    atoms = [(p / s, r) for p, r in atoms]
    return Gamble(
        "discrete + uniform band",
        atoms=atoms,
        bands=[(band_p / s, lo, hi)],
    )


GENERATORS = {1: gen_level_1, 2: gen_level_2, 3: gen_level_3, 4: gen_level_4}


def deal(rng: random.Random, level: int, leverage: float = 1.0) -> Gamble:
    lvl = level if level in GENERATORS else rng.choice([1, 2, 2, 3, 3, 4])
    for _ in range(80):
        g = GENERATORS[lvl](rng)
        # keep roughly one in five hands negative-edge, and don't let the
        # positive ones be so thin that the answer is a rounding error
        if g.edge() <= 0:
            if rng.random() < 0.20:
                return g
            continue
        f = g.kelly(leverage_cap=leverage)
        if f < 0.02:
            continue
        # mostly avoid hands whose answer is just "everything you have",
        # otherwise level 2 degenerates into pressing 100 over and over
        if f >= leverage * 0.999 and rng.random() < 0.8:
            continue
        return g
    return g


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def band_of_tolerance(g: Gamble, f_star: float, keep: float = 0.90, cap: float = 1.0):
    """
    The interval of stakes that retain at least `keep` of the optimal growth
    rate. Kelly is flat near its peak, so this band is embarrassingly wide —
    which is exactly the thing worth seeing.
    """
    if f_star <= 0:
        return (0.0, 0.0)
    target = keep * g.growth(f_star)
    lo, hi = 0.0, f_star
    for _ in range(80):
        mid = (lo + hi) / 2
        if g.growth(mid) < target:
            lo = mid
        else:
            hi = mid
    left = (lo + hi) / 2
    lo, hi = f_star, min(g.max_fraction() * (1 - 1e-9), cap)
    for _ in range(80):
        mid = (lo + hi) / 2
        if g.growth(mid) >= target:
            lo = mid
        else:
            hi = mid
    return (left, (lo + hi) / 2)


@dataclass
class HandResult:
    gamble: Gamble
    f_user: float
    f_star: float
    regret: float          # growth rate given up, per hand
    retained: float        # fraction of optimal growth kept
    outcome: float
    bankroll_before: float
    bankroll_after: float


# --------------------------------------------------------------------------
# The game loop
# --------------------------------------------------------------------------

RULE = "─" * 66


def ask_fraction(prompt: str) -> float | None:
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return None
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        raw = raw.rstrip("%")
        try:
            v = float(raw)
        except ValueError:
            print("    (enter a number: 0.25 or 25 both mean a quarter)")
            continue
        if v > 1.0:
            v /= 100.0
        if v < 0:
            print("    (no negative stakes in this version)")
            continue
        return v


def play(args) -> None:
    rng = random.Random(args.seed)
    bankroll = args.bankroll
    log: list[HandResult] = []

    print(RULE)
    print("  BET SIZING PRACTICE")
    print(f"  Bankroll ${bankroll:,.2f}.  {args.hands} hands.  'q' to stop early.")
    print("  Enter the fraction of your CURRENT bankroll you want to stake.")
    print(RULE)

    for i in range(1, args.hands + 1):
        g = deal(rng, args.level, leverage=args.leverage)
        print(f"\nHand {i}/{args.hands}   bankroll ${bankroll:,.2f}")
        print("  If you stake $1 on this bet:")
        print(g.describe())

        f = ask_fraction("  Stake what fraction? ")
        if f is None:
            print("\n  Stopping early.")
            break

        f_star = g.kelly(leverage_cap=args.leverage)
        g_star = g.growth(f_star)
        g_user = g.growth(f)
        regret = g_star - g_user
        retained = 1.0 if g_star <= 0 else (g_user / g_star if g_user > -math.inf else -math.inf)

        outcome = g.draw(rng)
        before = bankroll
        bankroll *= 1 + f * outcome
        log.append(HandResult(g, f, f_star, regret, retained, outcome, before, bankroll))

        print(f"\n  optimal stake      {f_star * 100:6.2f}%     you: {f * 100:.2f}%")
        if not args.terse:
            print(f"  edge (E[net])      {g.edge() * 100:6.2f}%  per unit staked")
            print(f"  worst loss         {g.worst_loss() * 100:6.2f}%  of the stake")
            if f_star > 0:
                lo, hi = band_of_tolerance(g, f_star, cap=args.leverage)
                print(f"  growth at optimum  {g_star * 100:6.3f}%  per hand")
                print(f"  90%-of-growth band {lo * 100:.1f}% – {hi * 100:.1f}%")
            else:
                print("  non-positive edge. the correct stake was 0.")

            if g_user == -math.inf:
                print("  you staked into ruin: growth rate is negative infinity.")
            elif f_star <= 0:
                if f > 1e-9:
                    print(f"  betting {f * 100:.0f}% cost you "
                          f"{-g_user * 100:.3f}% of growth per hand")
            elif retained <= 0:
                print(f"  you overbet past the break-even point. your growth rate on")
                print(f"  this hand is {g_user * 100:.3f}% per hand — negative. repeat it forever")
                print(f"  and the bankroll goes to zero no matter how good the bet was.")
            else:
                verdict = "over" if f > f_star else "under"
                print(f"  you kept {retained * 100:6.1f}% of the optimal growth rate ({verdict})")

        print(f"  → rolled {outcome * 100:+.1f}%.  bankroll ${bankroll:,.2f}")
        if bankroll < 1e-8:
            print("\n  Busted. That is the lesson.")
            break

    summarise(log, args.bankroll, bankroll)


def summarise(log: list[HandResult], start: float, end: float) -> None:
    if not log:
        return
    print("\n" + RULE)
    print("  SESSION SUMMARY")
    print(RULE)

    live = [h for h in log if h.f_star > 0]
    over = [h for h in live if h.f_user > h.f_star * 1.05]
    under = [h for h in live if h.f_user < h.f_star * 0.95]
    zero_hands = [h for h in log if h.f_star == 0]
    zero_hits = [h for h in zero_hands if h.f_user < 1e-9]

    total_regret = sum(h.regret for h in log if h.regret < math.inf)

    # Counterfactual: the same hands, the same realised outcomes, optimal stakes.
    # The draw doesn't depend on the stake, so this is the honest replay.
    kelly_path = start
    for h in log:
        kelly_path *= 1 + h.f_star * h.outcome
    kelly_median = start * math.exp(sum(h.gamble.growth(h.f_star) for h in log))
    realised_regret = (
        math.log(kelly_path / end) if kelly_path > 0 and end > 0 else math.inf
    )

    print(f"  hands played        {len(log)}")
    print(f"  bankroll            ${start:,.2f} → ${end:,.2f}")
    print(f"  overbet             {len(over)}   underbet {len(under)}")
    if zero_hands:
        print(f"  no-bet hands        {len(zero_hits)}/{len(zero_hands)} correctly passed")
    print(f"  growth lost, ex ante   {total_regret * 100:7.2f}%  (log points, summed)")
    if realised_regret < math.inf:
        print(f"  growth lost, realised  {realised_regret * 100:7.2f}%  (same dice)")
    print(f"\n  optimal play on THESE outcomes:  ${kelly_path:,.2f}   (you: ${end:,.2f})")
    print(f"  optimal play, median outcome:    ${kelly_median:,.2f}")
    if realised_regret < math.inf:
        luck = total_regret - realised_regret
        verdict = "the dice were kind to your errors" if luck > 0 else "the dice punished them"
        print(f"  luck component: {luck * 100:+.2f} log points — {verdict}")

    # Diagnostic: are you sizing off edge, or off edge-over-variance?
    if len(live) >= 5:
        fs = np.array([h.f_user for h in live])
        edges = np.array([h.gamble.edge() for h in live])
        stars = np.array([h.f_star for h in live])
        def corr(a, b):
            if a.std() < 1e-12 or b.std() < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])
        print(f"\n  your stake vs the edge          r = {corr(fs, edges):+.2f}")
        print(f"  your stake vs the optimal stake  r = {corr(fs, stars):+.2f}")
        print("  (the second should be much higher than the first. if it isn't,")
        print("   you are sizing off how good the bet looks, not off how much")
        print("   of you it can take away.)")

    print("\n  The bankroll number above is mostly noise over this many hands.")
    print("  The growth-lost number is the one that means anything.")
    print(RULE)


def selftest() -> None:
    """Two hands worked out by hand, to check the machinery."""
    a = Gamble("50/50 +20/-15", atoms=[(0.5, 0.20), (0.5, -0.15)])
    b = Gamble(
        "25% double / 50% U(0,1) / 25% forfeit",
        atoms=[(0.25, 1.0), (0.25, -1.0)],
        bands=[(0.5, 0.0, 1.0)],
    )
    c = Gamble("p=0.6 even money", atoms=[(0.6, 1.0), (0.4, -1.0)])
    for g, expected in ((a, 5 / 6), (b, 0.355), (c, 0.20)):
        got = g.kelly(leverage_cap=5.0)
        print(f"{g.label:38s} kelly = {got:.4f}   expected ≈ {expected:.4f}")
        assert abs(got - expected) < 0.01, "machinery is wrong"
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bet-sizing practice.")
    ap.add_argument("--hands", type=int, default=12)
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="max stake as a fraction of bankroll; >1 allows borrowing")
    ap.add_argument("--terse", action="store_true", help="hide the post-hand breakdown")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    try:
        play(args)
    except KeyboardInterrupt:
        print("\n  Out.")
        sys.exit(0)


if __name__ == "__main__":
    main()
