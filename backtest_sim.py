# backtest_sim.py: would the simulator have got races right that already happened?
# Pace and retirement risk are built from earlier rounds only, then the race is
# simulated from its real grid and compared against the actual result.

import warnings
import numpy as np
import pandas as pd

import dnf_risk
import pace_model
import race_sim

warnings.filterwarnings("ignore")
K = 5        # shrinkage strength, picked in dnf_risk.py


def predict_pace(feats, rnd):
    """Pace prediction for one round, trained on earlier rounds only."""
    tr = feats[(feats["round"] < rnd)].dropna(subset=["pace_gap_pct"])
    te = feats[feats["round"] == rnd].copy()
    ms = [m for name, m in pace_model.models().items() if name[0] in "234"]
    te["pace_gap_pct"] = np.mean([m.fit(tr[pace_model.FEATURES], tr["pace_gap_pct"]).predict(te[pace_model.FEATURES])
                                  for m in ms], axis=0)
    return te


def run(params=None, n=2000, rounds=range(6, 15), verbose=True):
    events = dnf_risk.build_events(dnf_risk.load_results())
    events["round"] = events["race"].str.extract(r"R(\d+)$").astype(float)
    raw = pd.read_csv("data/pace_dataset_2026.csv")
    feats = pace_model.make_features(raw)

    out = []
    for rnd in rounds:
        train = events[(events["year"] < 2026) | (events["round"] < rnd)]
        lineup = events[events["race"] == f"2026-R{rnd}"][["driver", "team", "supplier", "category"]]
        cars = dnf_risk.car_probabilities(train, lineup, K)
        pace = predict_pace(feats, rnd)[["driver", "pace_gap_pct", "grid"]]
        cars = cars.merge(pace, on="driver").reset_index(drop=True)
        cars["grid"] = cars["grid"].replace(0, np.nan).fillna(99)
        laps = LAPS[rnd]
        finish = race_sim.simulate(cars, laps, n=n, seed=rnd, params=params)

        cars["exp_finish"] = finish.mean(axis=0)
        cars["win_prob"] = (finish == 1).mean(axis=0)
        cars["pred_rank"] = cars["exp_finish"].rank(method="first")
        actual = ACTUAL[rnd].reindex(cars["driver"]).values
        cars["actual"] = actual
        cars["grid_rank"] = cars["grid"].rank(method="first")
        started = cars["actual"].notna()
        mae_sim = np.abs(cars.loc[started, "pred_rank"] - cars.loc[started, "actual"]).mean()
        mae_grid = np.abs(cars.loc[started, "grid_rank"] - cars.loc[started, "actual"]).mean()
        win = cars.loc[cars["actual"] == 1, "driver"].values
        out.append(dict(round=rnd, mae_sim=mae_sim, mae_grid=mae_grid,
                        winner_pred=cars.loc[cars["pred_rank"] == 1, "driver"].values[0],
                        winner_actual=win[0] if len(win) else None,
                        winner_prob=cars.loc[cars["driver"].isin(win), "win_prob"].sum()))
    out = pd.DataFrame(out)
    if verbose:
        print(out.round(3).to_string(index=False))
        print(f"\nAverage places off, simulator: {out.mae_sim.mean():.2f} | grid order: {out.mae_grid.mean():.2f}")
        print(f"Winner called right: {(out.winner_pred == out.winner_actual).sum()}/{len(out)} | "
              f"avg probability given to the actual winner: {out.winner_prob.mean():.1%}")
    return out


# Race distance and real finishing positions, straight from the results.
def load_truth():
    import fastf1
    fastf1.Cache.enable_cache("cache"); fastf1.set_log_level("ERROR")
    laps, actual = {}, {}
    for rnd in range(1, 15):
        s = fastf1.get_session(2026, rnd, "R"); s.load(laps=False, telemetry=False, weather=False, messages=False)
        r = s.results
        laps[rnd] = int(r["Laps"].max())
        actual[rnd] = r.set_index("Abbreviation")["Position"]
    return laps, actual


LAPS, ACTUAL = load_truth()

if __name__ == "__main__":
    run()
