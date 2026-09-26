# tyre_pitstop.py: tyre degradation and pit stop cost, measured from Baku 2021-2025
# Produces: per-compound wear rate (s/lap of tyre age), pit stop loss (s), and a table
# of real strategies (compound sequence + stop laps) to sample from in the Monte Carlo.

import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

YEARS = [2021, 2022, 2023, 2024, 2025]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: TYRE DEGRADATION PER COMPOUND
# ════════════════════════════════════════════════════════════════════════════
# lap_time = driver_pace + compound_offset + deg_rate[compound] * tyre_age + fuel_burn * lap_number + noise
# Fitting one deg_rate PER COMPOUND (not just one overall) is what tells us how much
# faster a soft tyre falls off than a hard one at Baku.

def clean_laps(laps):
    l = laps.copy()
    l = l[(l["TrackStatus"] == "1") & (l["LapNumber"] > 1)]
    l = l[l["PitInTime"].isna() & l["PitOutTime"].isna()]
    l = l[l["LapTime"].notna() & l["TyreLife"].notna() & ~l["Deleted"].fillna(False).astype(bool)]
    l["t"] = l["LapTime"].dt.total_seconds()
    l = l[l["t"] < l.groupby("Driver")["t"].transform("min") * 1.08]
    return l


def fit_degradation(all_laps):
    l = pd.concat(all_laps, ignore_index=True)
    compounds = sorted(l["Compound"].unique())
    driver_race = l["year"].astype(str) + l["Driver"]           # a driver's pace differs race to race
    X = pd.get_dummies(driver_race).astype(float)
    n_dr = X.shape[1]
    for c in compounds:
        X[f"age_{c}"] = np.where(l["Compound"] == c, l["TyreLife"].astype(float), 0.0)
    X["lap_no"] = l["LapNumber"].astype(float)
    coef, *_ = np.linalg.lstsq(X.values, l["t"].values, rcond=None)
    rates = pd.Series(coef[n_dr:n_dr + len(compounds)], index=[f"{c}" for c in compounds])
    return rates, l


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: PIT STOP LOSS
# ════════════════════════════════════════════════════════════════════════════
# Total time given up by pitting = (in-lap time + out-lap time) - 2 x that driver's
# normal lap time. This bundles the deceleration into the pits, the stationary
# stop, the pit-lane speed limit and the re-acceleration into one number.

def pit_loss(all_laps):
    losses = []
    for l in all_laps:
        laps = l.copy()
        laps["t"] = laps["LapTime"].dt.total_seconds()
        normal = laps[(laps["PitInTime"].isna()) & (laps["PitOutTime"].isna()) & (laps["TrackStatus"] == "1")]
        normal_pace = normal.groupby("Driver")["t"].median()
        for drv, g in laps.groupby("Driver"):
            if drv not in normal_pace.index:
                continue
            base = normal_pace[drv]
            in_lap = g[g["PitInTime"].notna()]
            out_lap = g[g["PitOutTime"].notna()]
            for _, r in in_lap.iterrows():
                nxt = out_lap[out_lap["LapNumber"] == r["LapNumber"] + 1]
                if pd.notna(r["LapTime"]) and len(nxt) and pd.notna(nxt.iloc[0]["LapTime"]):
                    total = r["LapTime"].total_seconds() + nxt.iloc[0]["LapTime"].total_seconds()
                    loss = total - 2 * base
                    if 0 < loss < 40:      # discard SC-affected or broken stops
                        losses.append(loss)
    return pd.Series(losses)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: REAL STRATEGIES (to sample from in the Monte Carlo)
# ════════════════════════════════════════════════════════════════════════════

def strategies(raw_laps_by_year):
    rows = []
    for year, laps in raw_laps_by_year.items():
        for (drv), g in laps.sort_values("LapNumber").groupby("Driver"):
            stints = g.groupby("Stint").agg(compound=("Compound", "first"),
                                             start_lap=("LapNumber", "min"),
                                             end_lap=("LapNumber", "max"))
            if len(stints) < 1 or stints["compound"].isna().any():
                continue
            stops = stints["start_lap"].iloc[1:].astype(int).tolist()
            rows.append(dict(year=year, driver=drv, n_stops=len(stints) - 1,
                             compounds=tuple(stints["compound"]), stop_laps=tuple(stops),
                             race_laps=int(g["LapNumber"].max())))
    return pd.DataFrame(rows)


def load_all():
    laps_by_year, cleaned = {}, []
    for y in YEARS:
        s = fastf1.get_session(y, "Azerbaijan", "R")
        s.load(telemetry=False, weather=False, messages=False)
        laps_by_year[y] = s.laps.copy()
        c = clean_laps(s.laps); c["year"] = y
        cleaned.append(c)
    return laps_by_year, cleaned


if __name__ == "__main__":
    laps_by_year, cleaned = load_all()

    rates, used = fit_degradation(cleaned)
    print("Baku tyre degradation, seconds lost per lap of tyre age:")
    print(rates.round(3).to_string())
    print(f"(fitted on {len(used)} clean green-flag laps, Baku 2021-2025)")

    losses = pit_loss([l for l in laps_by_year.values()])
    print(f"\nPit stop loss: median {losses.median():.2f}s, mean {losses.mean():.2f}s, "
          f"n={len(losses)}, range {losses.min():.1f}-{losses.max():.1f}s")

    strat = strategies(laps_by_year)
    print(f"\nStrategies seen ({len(strat)} driver-races):")
    print(strat.groupby("n_stops").size().to_string())
    print("\nMost common compound sequences:")
    print(strat["compounds"].value_counts().head(8).to_string())

    rates.to_csv("data/baku_tyre_degradation.csv", header=["sec_per_lap_age"])
    losses.to_csv("data/baku_pit_losses.csv", index=False, header=["loss_seconds"])
    strat.to_pickle("data/baku_strategies.pkl")
