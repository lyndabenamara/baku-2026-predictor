# race_sim.py: the race simulator.
# Runs the race lap by lap for every car, and runs N races at once with numpy.

import numpy as np

# Everything tunable lives here. Measured values say where they came from,
# the rest are assumptions.
PARAMS = dict(
    base_lap=95.0,          # seconds, only the ratio to it matters
    noise_sd=0.35,          # lap to lap wobble per car
    day_form_sd=0.15,       # how far off its expected pace a car's whole race can be
    pit_loss_sc=10.0,       # pitting under a safety car, roughly 45% of a green flag stop
    start_gap=0.25,         # seconds between grid slots at the start
    min_gap=0.30,           # how close a blocked car sits behind the one ahead
    dirty_air=0.90,         # seconds a lap lost sitting in someone's wake
    dirty_air_range=1.5,    # you count as in traffic within this many seconds
    overtake_thresh=0.20,   # pace advantage needed to get past, s/lap
    p_safety_car=0.6,       # Baku had a full safety car in 3 of the last 5 races
    sc_laps=3,              # how long the field stays neutralised
    sc_gap=0.5,             # gap between cars once the safety car bunches them up
    early_share=0.4,        # share of crashes that happen in the first few laps
    pit_loss=22.9,          # median of 78 real Baku stops
    deg=dict(HARD=0.020, MEDIUM=0.013, SOFT=0.039),   # seconds lost per lap of tyre age
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: WHO RETIRES, AND WHEN
# ════════════════════════════════════════════════════════════════════════════
# Roll the dice separately for each cause. Engine, mechanical and tyre failures
# can hit on any lap. Crashes are weighted towards the opening laps.
# Anything counted in a car's DNF probability has to be rolled here too, or the
# DNF column would describe races that never happened.

CAUSES = [("p_pu", False), ("p_mech", False), ("p_tyre", False), ("p_incident", True)]


def retirement_laps(rng, cars, n, laps, p):
    C = len(cars)
    fail = np.full((n, C), np.inf)
    for col, early in CAUSES:
        if col not in cars:
            continue
        happens = rng.random((n, C)) < cars[col].values
        lap = rng.integers(1, laps + 1, size=(n, C)).astype(float)
        if early:
            is_early = rng.random((n, C)) < p["early_share"]
            lap = np.where(is_early, rng.integers(1, 4, size=(n, C)), lap)
        fail = np.minimum(fail, np.where(happens, lap, np.inf))
    return fail


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: RUN THE RACES
# ════════════════════════════════════════════════════════════════════════════
# cars needs: pace_gap_pct, grid, and the DNF probability columns.
# strategies is optional, and without it tyre wear and pit stops are skipped.
# Returns the finishing position of every car in every race, retirees at the back.

def build_tyre_tables(rng, strategies, n, C, laps, deg):
    """Builds one tyre cost and pit lap table per strategy and then looks them up by
    index, since there are only about 100 strategies but thousands of cars."""
    idx = rng.integers(0, len(strategies), size=(n, C))
    S = len(strategies)
    cost = np.zeros((S, laps + 1))
    pit = np.zeros((S, laps + 1), dtype=bool)
    for s, row in enumerate(strategies.itertuples()):
        scale = laps / row.race_laps
        stops = sorted(set(max(1, min(laps - 1, round(x * scale))) for x in row.stop_laps))
        bounds = [0] + stops + [laps]
        for seg, comp in enumerate(row.compounds[: len(bounds) - 1]):
            lo, hi = bounds[seg], bounds[seg + 1]
            a = np.arange(1, hi - lo + 1)
            cost[s, lo + 1: hi + 1] = a * deg.get(comp, 0.02)
        for stop in stops:
            pit[s, stop] = True
    return idx, cost, pit


def simulate(cars, laps, n=2000, seed=0, params=None, strategies=None, return_retirements=False):
    p = {**PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)
    C = len(cars)
    pace = p["base_lap"] * (1 + cars["pace_gap_pct"].values / 100)
    # Whether each car is having a good or bad day, redrawn for every race.
    # Erratic drivers swing further than consistent ones.
    spread = cars["noise_mult"].values if "noise_mult" in cars else 1.0
    day_form = rng.normal(0, p["day_form_sd"], size=(n, C)) * spread
    grid_rank = cars["grid"].rank(method="first").values - 1
    t = np.tile(grid_rank * p["start_gap"], (n, 1)).astype(float)                 # race clock per car
    fail_lap = retirement_laps(rng, cars, n, laps, p)
    laps_done = np.zeros((n, C))

    has_sc = rng.random(n) < p["p_safety_car"]
    sc_start = np.where(rng.random(n) < 0.25, rng.integers(1, 4, n), rng.integers(4, max(laps - 4, 5), n))

    if strategies is not None and len(strategies):
        strat_idx, cost_tbl, pit_tbl = build_tyre_tables(rng, strategies, n, C, laps, p["deg"])
    else:
        # No strategies given, so use empty tables: no tyre wear and nobody pits
        strat_idx = np.zeros((n, C), dtype=int)
        cost_tbl = np.zeros((1, laps + 1))
        pit_tbl = np.zeros((1, laps + 1), dtype=bool)

    for lap in range(1, laps + 1):
        alive = fail_lap > lap - 1              # still running at the start of this lap
        lap_time = pace + day_form + rng.normal(0, p["noise_sd"], size=(n, C))
        lap_time = lap_time + cost_tbl[strat_idx, lap]               # tyre wear so far this stint
        under_sc = (has_sc & (lap >= sc_start) & (lap < sc_start + p["sc_laps"]))[:, None]
        # Behind the safety car everyone laps at the same slow pace, but a stop still
        # costs you something, just much less than it would under green
        lap_time = np.where(under_sc, p["base_lap"] * 1.35, lap_time)
        pit_cost = np.where(under_sc, p["pit_loss_sc"], p["pit_loss"])
        lap_time = lap_time + np.where(pit_tbl[strat_idx, lap], pit_cost, 0.0)
        prev_order = np.argsort(np.where(alive, t, 1e9 + np.arange(C)), axis=1)
        new_t = np.where(alive, t + lap_time, t)
        # Sitting in someone's wake costs you time, and you can only get past if
        # you're clearly quicker. This is what makes fighting through the field slow.
        rows = np.arange(n)
        for j in range(1, C):
            c, a = prev_order[:, j], prev_order[:, j - 1]
            both = alive[rows, c] & alive[rows, a]
            gap = new_t[rows, c] - new_t[rows, a]
            in_wake = both & (gap < p["dirty_air_range"]) & ~under_sc[rows, 0]
            new_t[rows, c] = new_t[rows, c] + np.where(in_wake, p["dirty_air"], 0.0)
            advantage = lap_time[rows, a] - lap_time[rows, c]
            blocked = both & (new_t[rows, c] < new_t[rows, a] + p["min_gap"]) & (advantage < p["overtake_thresh"])
            new_t[rows, c] = np.where(blocked, new_t[rows, a] + p["min_gap"], new_t[rows, c])
        # Safety car ends: bunch the field, order preserved.
        ending = has_sc & (lap == sc_start + p["sc_laps"] - 1)
        if ending.any():
            key = np.where(alive, new_t, 1e9 + np.arange(C))
            order = np.argsort(key, axis=1)
            rank = np.argsort(order, axis=1)
            lead = np.min(np.where(alive, new_t, np.inf), axis=1, keepdims=True)
            new_t = np.where(ending[:, None] & alive, lead + rank * p["sc_gap"], new_t)
        t = new_t
        laps_done = np.where(alive, lap, laps_done)

    score = laps_done * 1e6 - t                # more laps first, then lower race time
    order = np.argsort(-score, axis=1)
    finish = np.empty_like(order)
    np.put_along_axis(finish, order, np.arange(1, C + 1)[None, :].repeat(n, 0), axis=1)
    if return_retirements:
        return finish, np.isfinite(fail_lap)   # True = this car retired in this race
    return finish
