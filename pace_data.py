# pace_data.py: builds the race-pace dataset for the Baku 2026 predictor
# One row per driver per 2026 race. Target = pace_gap_pct (how much slower than the
# fastest car the driver's clean racing pace was). Features = things known BEFORE the race.

import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

SUPPLIER = {
    "Mercedes": "Mercedes", "McLaren": "Mercedes", "Williams": "Mercedes", "Alpine": "Mercedes",
    "Ferrari": "Ferrari", "Haas F1 Team": "Ferrari", "Cadillac": "Ferrari",
    "Red Bull Racing": "Red Bull-Ford", "Racing Bulls": "Red Bull-Ford",
    "Aston Martin": "Honda", "Audi": "Audi",
}
MIN_LAPS = 8          # fewer clean laps than this = not enough evidence for a pace number
SLOW_CUTOFF = 1.07    # ignore laps >7% slower than the race's fastest clean lap (traffic, damage, mistakes)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: CLEAN RACING LAPS
# ════════════════════════════════════════════════════════════════════════════
# Only keep laps that actually show pace: green flag, not lap 1, no pit in or
# out laps, dry tyres, valid time.

def clean_laps(laps):
    l = laps.copy()
    l = l[(l["TrackStatus"] == "1") & (l["LapNumber"] > 1)]
    l = l[l["PitInTime"].isna() & l["PitOutTime"].isna()]
    l = l[l["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    l = l[l["LapTime"].notna() & l["TyreLife"].notna() & ~l["Deleted"].fillna(False).astype(bool)]
    l["t"] = l["LapTime"].dt.total_seconds()
    l = l[l["t"] < l["t"].min() * SLOW_CUTOFF]
    return l[["Driver", "LapNumber", "Compound", "TyreLife", "t"]]


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: STRIP OUT TYRE AGE AND FUEL
# ════════════════════════════════════════════════════════════════════════════
# Lap time is driver + compound + tyre age + fuel + noise. Fitting all of that at
# once gives you each driver's own pace, so nobody gets judged on old tyres or a
# heavy fuel load.

def driver_pace(l):
    X = pd.get_dummies(l["Driver"]).astype(float)
    X = X.join(pd.get_dummies(l["Compound"], drop_first=True).astype(float))
    X["tyre_age"] = l["TyreLife"].astype(float).values
    X["lap_no"] = l["LapNumber"].astype(float).values
    coef, *_ = np.linalg.lstsq(X.values, l["t"].values, rcond=None)
    pace = pd.Series(coef[: l["Driver"].nunique()], index=X.columns[: l["Driver"].nunique()])
    counts = l.groupby("Driver").size()
    pace = pace[counts[counts >= MIN_LAPS].index]
    return (100 * (pace - pace.min()) / pace.min()), counts


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: QUALIFYING FEATURES
# ════════════════════════════════════════════════════════════════════════════
# Gap to pole as a percentage, plus top speed on their fastest lap.

def quali_features(q):
    laps = q.laps.pick_quicklaps(1.15)
    best = laps.groupby("Driver").apply(lambda d: d.loc[d["LapTime"].idxmin()])
    t = best["LapTime"].dt.total_seconds()
    return pd.DataFrame({"quali_gap_pct": 100 * (t - t.min()) / t.min(), "quali_speed_trap": best["SpeedST"]})


def build():
    rows = []
    for rnd in range(1, 15):
        r = fastf1.get_session(2026, rnd, "R"); r.load(telemetry=False, weather=False, messages=False)
        q = fastf1.get_session(2026, rnd, "Q"); q.load(telemetry=False, weather=False, messages=False)
        pace, counts = driver_pace(clean_laps(r.laps))
        qf = quali_features(q)
        res = r.results.set_index("Abbreviation")
        for d in res.index:
            rows.append(dict(
                round=rnd, driver=d, team=res.loc[d, "TeamName"], grid=res.loc[d, "GridPosition"],
                clean_laps=int(counts.get(d, 0)), pace_gap_pct=pace.get(d, np.nan),
                quali_gap_pct=qf["quali_gap_pct"].get(d, np.nan),
                quali_speed_trap=qf["quali_speed_trap"].get(d, np.nan)))
        print(f"round {rnd} done", flush=True)
    df = pd.DataFrame(rows)
    df["supplier"] = df["team"].map(SUPPLIER)
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv("data/pace_dataset_2026.csv", index=False)
    ok = df.dropna(subset=["pace_gap_pct", "quali_gap_pct"])
    print(f"\nrows: {len(df)} | usable (pace + quali): {len(ok)}")
    print("missing pace (too few clean laps):", df.pace_gap_pct.isna().sum(),
          "| missing quali:", df.quali_gap_pct.isna().sum())
    print("\ncorrelation quali gap vs race pace gap (whole season):",
          round(ok.quali_gap_pct.corr(ok.pace_gap_pct), 3))
    print("\nfastest 3 by race pace, each round:")
    for rnd, g in ok.groupby("round"):
        top = g.nsmallest(3, "pace_gap_pct")
        print(f"  R{rnd:<2}", ", ".join(f"{d} {p:.2f}%" for d, p in zip(top.driver, top.pace_gap_pct)),
              "| median clean laps:", int(g.clean_laps.median()))
