# predict_baku.py: main script for the Baku 2026 prediction.
# Combines pace, DNF risk, tyres and pit stops, then runs 10,000 simulated races.
#
# To run it for a race weekend:
#   1. Update the qualifying CSVs in data/
#   2. Check LINEUP, GRID_PENALTY_BACK, COMPROMISED_QUALI, UPGRADE_FLAG, PU_SPEC_ADUO2
#   3. python predict_baku.py

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dnf_risk
import pace_model
import race_sim
import baku_features

warnings.filterwarnings("ignore")

N_SIMS = 10_000
BAKU_LAPS = 51


# ════════════════════════════════════════════════════════════════════════════
# RACE WEEKEND INPUTS
# ════════════════════════════════════════════════════════════════════════════

# Hadjar is back at Red Bull for this race, so Lawson returns to Racing Bulls
# and Tsunoda drops out. Checked against the FP3 entry list.
LINEUP = {
    "RUS": "Mercedes", "ANT": "Mercedes",
    "VER": "Red Bull Racing", "HAD": "Red Bull Racing",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "NOR": "McLaren", "PIA": "McLaren",
    "ALB": "Williams", "SAI": "Williams",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "BEA": "Haas F1 Team", "OCO": "Haas F1 Team",
    "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "HUL": "Audi", "BOR": "Audi",
    "GAS": "Alpine", "COL": "Alpine",
    "PER": "Cadillac", "BOT": "Cadillac",
}

# Qualifying, rebuilt from the session lap data. The grid comes from the
# classification and not from sorting the times, because everyone who reached Q3
# starts ahead of everyone knocked out in Q2 even if the Q2 car was quicker.
_Q = pd.read_csv("data/baku2026_qualifying.csv")
QUALI_TIME = dict(zip(_Q.driver, _Q.time_s))
QUALI_SPEED_TRAP = dict(zip(_Q.driver, _Q.speed_trap))
GRID = dict(zip(_Q.driver, _Q.P))

# Anyone here gets dropped to the back of the grid. Empty this weekend: Leclerc's
# engine was repaired rather than replaced, so Ferrari stayed inside his allocation.
GRID_PENALTY_BACK = set()

# Drivers whose qualifying lap says nothing useful about their car, because they
# crashed or hit a problem. Their lap gets swapped for their practice pace, but they
# still start where they qualified. Antonelli put it in the wall in Q1 here.
COMPROMISED_QUALI = {"ANT"}

# Teams that brought a big upgrade this weekend. For these the season-long form is
# describing the old car, so lean on this weekend's pace instead.
UPGRADE_FLAG = {"Williams": True, "Audi": True}

# Ferrari drivers on the newer ADUO2 engine. It's worth about 18hp between
# 260 and 310 km/h, which is most of Baku's main straight.
PU_SPEC_ADUO2 = {"HAM", "LEC"}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD THE CAR TABLE
# ════════════════════════════════════════════════════════════════════════════

def merit_grid():
    """Qualifying order straight from the lap times, before any penalty."""
    return {d: i + 1 for i, d in enumerate(sorted(LINEUP, key=lambda d: QUALI_TIME[d]))}


def race_grid():
    """Where everyone actually starts, with penalised drivers moved to the back."""
    grid = dict(GRID) if GRID else merit_grid()
    if GRID_PENALTY_BACK:
        order = sorted(grid, key=lambda d: grid[d])
        front = [d for d in order if d not in GRID_PENALTY_BACK]
        back = [d for d in order if d in GRID_PENALTY_BACK]
        grid = {d: i + 1 for i, d in enumerate(front + back)}
    return grid


def build_cars():
    lineup_df = pd.DataFrame({"driver": list(LINEUP), "team": list(LINEUP.values())})
    lineup_df["supplier"] = lineup_df["team"].map(dnf_risk.SUPPLIER)

    # DNF risk, trained on this season plus every Baku race since 2021
    events = dnf_risk.build_events(dnf_risk.load_results())
    cars = dnf_risk.car_probabilities(events, lineup_df.assign(category=np.nan), k=5)
    cars["p_tyre"] = 2 / 100   # 2 tyre failures in 100 Baku starts
    cars["p_dnf"] = 1 - (1 - cars.p_dnf) * (1 - cars.p_tyre)

    # Pace, from the ensemble trained on the season so far
    raw = pd.read_csv("data/pace_dataset_2026.csv")
    feats = pace_model.make_features(raw)
    quali = pd.DataFrame({
        "driver": list(LINEUP),
        "quali_gap_pct": [100 * (QUALI_TIME[d] - min(QUALI_TIME.values())) / min(QUALI_TIME.values())
                          for d in LINEUP],
        "quali_speed_trap": [QUALI_SPEED_TRAP[d] for d in LINEUP],
    })
    # A crashed qualifying lap tells you nothing about the car, so swap it for that
    # driver's practice pace. Season form is the fallback if there's no practice data.
    if COMPROMISED_QUALI:
        practice = pd.read_csv("data/baku2026_practice.csv", index_col=0)["best_practice_gap_pct"]
        season = raw.groupby("driver")["quali_gap_pct"].median()
        is_bad = quali["driver"].isin(COMPROMISED_QUALI)
        sub = quali.loc[is_bad, "driver"].map(practice)
        sub = sub.fillna(quali.loc[is_bad, "driver"].map(season)).fillna(quali["quali_gap_pct"].median())
        quali.loc[is_bad, "quali_gap_pct"] = sub
        quali.loc[is_bad, "quali_speed_trap"] = quali["quali_speed_trap"].median()

    quali["speed_rel"] = quali["quali_speed_trap"] - quali["quali_speed_trap"].median()
    # The grid feature here is ranked from pace, not from the real starting grid. A
    # penalty or a crash changes where you start, not how quick the car is.
    quali["grid"] = quali["quali_gap_pct"].rank(method="first")
    team_form = raw.groupby("team")["pace_gap_pct"].mean()
    quali["team_form"] = quali["driver"].map(LINEUP).map(team_form).fillna(raw["pace_gap_pct"].mean())
    supplier_speed = feats.groupby("supplier")["speed_rel"].mean() if "speed_rel" in feats else pd.Series(dtype=float)
    quali["supplier_speed"] = quali["driver"].map(LINEUP).map(dnf_risk.SUPPLIER).map(supplier_speed).fillna(0)

    tr = feats.dropna(subset=["pace_gap_pct"])
    ms = [m for name, m in pace_model.models().items() if name[0] in "234"]
    quali["pace_gap_pct"] = np.mean(
        [m.fit(tr[pace_model.FEATURES], tr["pace_gap_pct"]).predict(quali[pace_model.FEATURES]) for m in ms], axis=0)

    cars = cars.merge(quali[["driver", "pace_gap_pct", "quali_gap_pct"]], on="driver")
    # Only now does the real starting grid come in, penalties and all
    cars["grid"] = cars["driver"].map(race_grid())

    # Small nudge for drivers who historically do better or worse than their grid
    # slot at this track specifically
    spec = pd.read_csv("data/baku_specialist_scores.csv", index_col=0)["specialist_score"]
    cars["specialist_score"] = cars["driver"].map(spec).fillna(0)
    cars["pace_gap_pct"] -= cars["specialist_score"] * 0.05

    # ADUO2 engine, roughly 0.2s a lap here
    base_lap = 95.0
    cars.loc[cars["driver"].isin(PU_SPEC_ADUO2), "pace_gap_pct"] -= 100 * 0.2 / base_lap

    # For upgraded teams the season form describes the old car, so blend it halfway
    # towards what they actually did in qualifying
    upgraded = cars["team"].map(UPGRADE_FLAG).fillna(False)
    cars.loc[upgraded, "pace_gap_pct"] = (
        0.5 * cars.loc[upgraded, "pace_gap_pct"] + 0.5 * cars.loc[upgraded, "quali_gap_pct"])

    # Erratic drivers get a wider spread of outcomes
    cars["noise_mult"] = baku_features.variance_multiplier(cars["p_incident"])

    return cars.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: RUN THE MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════

def run(cars):
    """Every race in one call so each one gets its own roll of form, retirements,
    strategy and safety car. Returns (finish, retired)."""
    strat = pd.read_pickle("data/baku_strategies.pkl")
    deg = pd.read_csv("data/baku_tyre_degradation.csv", index_col=0)["sec_per_lap_age"].to_dict()
    pit_loss = pd.read_csv("data/baku_pit_losses.csv")["loss_seconds"].median()
    return race_sim.simulate(cars, BAKU_LAPS, n=N_SIMS, seed=42,
                             params=dict(pit_loss=pit_loss, deg=deg),
                             strategies=strat, return_retirements=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: TURN THE SIMULATIONS INTO RESULTS
# ════════════════════════════════════════════════════════════════════════════

def summarize(cars, finish, retired):
    """Counts everything from the simulated races, so the DNF column and the
    finishing positions always describe the same set of races."""
    cars = cars.copy()
    cars["exp_finish"] = finish.mean(0)
    finished = ~retired
    cars["win_%"] = 100 * ((finish == 1) & finished).mean(0)
    cars["top3_%"] = 100 * ((finish <= 3) & finished).mean(0)
    cars["points_%"] = 100 * ((finish <= 10) & finished).mean(0)
    cars["dnf_%"] = 100 * retired.mean(0)
    order = cars.sort_values("exp_finish").reset_index(drop=True)
    order.insert(0, "predicted_position", range(1, len(order) + 1))
    return order


def charts(order, finish, drivers):
    fig, ax = plt.subplots(figsize=(10, 7))
    top = order.head(12).iloc[::-1]
    ax.barh(top["driver"], top["win_%"], color="#c00000", label="Win %")
    ax.barh(top["driver"], top["top3_%"] - top["win_%"], left=top["win_%"], color="#e6a4a4", label="Top 3 % (extra)")
    ax.set_xlabel("Probability (%)"); ax.set_title("Baku 2026 - win / top-3 probability"); ax.legend()
    fig.tight_layout(); fig.savefig("data/baku_probability_chart.png", dpi=150)
    plt.close(fig)

    # Heatmap: how often each driver lands in each position
    C = finish.shape[1]
    heat = np.zeros((C, C))
    for j in range(len(drivers)):
        counts = np.bincount(finish[:, j], minlength=C + 1)[1:]
        heat[j] = counts / finish.shape[0]
    row_order = order["driver"].map({d: i for i, d in enumerate(drivers)}).values
    heat = heat[row_order]

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    im = ax2.imshow(heat, aspect="auto", cmap="Reds")
    ax2.set_yticks(range(C)); ax2.set_yticklabels(order["driver"])
    ax2.set_xticks(range(C)); ax2.set_xticklabels([str(p) for p in range(1, C + 1)])
    ax2.set_xlabel("Finishing position"); ax2.set_title("Baku 2026 - finishing position probability")
    fig2.colorbar(im, ax=ax2, label="Probability")
    fig2.tight_layout(); fig2.savefig("data/baku_heatmap.png", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    cars = build_cars()
    finish, retired = run(cars)
    order = summarize(cars, finish, retired)
    charts(order, finish, cars["driver"].tolist())
    pd.DataFrame(finish, columns=cars["driver"]).to_csv("data/baku_sim_raw.csv", index=False)
    order.to_csv("data/baku_prediction.csv", index=False)

    print("=" * 70)
    print("  BAKU 2026 - PREDICTED RESULT")
    print(f"  Real qualifying ({len(_Q)} drivers) | {N_SIMS:,} simulated races")
    print("=" * 70)
    cols = ["predicted_position", "driver", "team", "win_%", "top3_%", "points_%", "dnf_%"]
    print(order[cols].round(1).to_string(index=False))
    if GRID_PENALTY_BACK:
        print(f"\n  Grid penalty applied (starts at the back): {', '.join(sorted(GRID_PENALTY_BACK))}")
    if COMPROMISED_QUALI:
        print(f"  Qualifying lap treated as unrepresentative: {', '.join(sorted(COMPROMISED_QUALI))}")
