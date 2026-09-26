# baku_features.py: two features built from history: a Baku specialist score
# and a variance multiplier for erratic drivers.

import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")


# ════════════════════════════════════════════════════════════════════════════
# BAKU SPECIALIST SCORE
# ════════════════════════════════════════════════════════════════════════════
# Does this driver usually finish better or worse than they qualified, here
# specifically? It's the average of (grid - finish) across their Baku starts,
# shrunk towards 0 for anyone without much history at the track.

def specialist_scores(k=3):
    rows = []
    for y in range(2021, 2026):
        s = fastf1.get_session(y, "Azerbaijan", "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        r = s.results
        rows.append(r[["Abbreviation", "GridPosition", "Position"]].assign(year=y))
    df = pd.concat(rows, ignore_index=True)
    df["delta"] = df["GridPosition"] - df["Position"]     # positive = gained places at Baku
    g = df.groupby("Abbreviation")["delta"].agg(["mean", "count"])
    return ((g["mean"] * g["count"]) / (g["count"] + k)).rename("specialist_score")   # shrunk toward 0


# ════════════════════════════════════════════════════════════════════════════
# DRIVER VARIANCE
# ════════════════════════════════════════════════════════════════════════════
# Erratic drivers should get a wider spread of outcomes, not just a higher crash
# number. This scales how much their race swings, reusing p_incident rather than
# inventing a second metric for the same thing.

def variance_multiplier(p_incident, lo=0.7, hi=1.6):
    avg = p_incident.mean()
    mult = 0.7 + (p_incident / avg) * 0.5
    return mult.clip(lo, hi)


if __name__ == "__main__":
    sc = specialist_scores().sort_values(ascending=False)
    print("Baku specialist score (+ = gains places at Baku, shrunk toward 0 for thin history)")
    print(sc.round(2).to_string())
    sc.to_csv("data/baku_specialist_scores.csv")
