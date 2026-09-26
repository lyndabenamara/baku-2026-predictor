# pace_model.py: checks whether the ML actually beats just using qualifying.
# For each round it trains on earlier rounds only, then predicts that round.

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: FEATURES
# ════════════════════════════════════════════════════════════════════════════
# All of these are known before the race starts:
#   quali_gap_pct   gap to pole in qualifying
#   speed_rel       top speed against the field median, so straight line pace
#   team_form       the team's average race pace over earlier rounds only
#   supplier_speed  the engine supplier's average top speed, earlier rounds only
#   grid            starting position

def make_features(df):
    d = df.sort_values(["round", "driver"]).copy()
    d["quali_gap_pct"] = d["quali_gap_pct"].fillna(d.groupby("round")["quali_gap_pct"].transform("max"))
    d["speed_rel"] = d["quali_speed_trap"] - d.groupby("round")["quali_speed_trap"].transform("median")
    d["speed_rel"] = d["speed_rel"].fillna(0)
    d["team_form"] = np.nan
    d["supplier_speed"] = np.nan
    for rnd in sorted(d["round"].unique()):
        past = d[d["round"] < rnd]
        if past.empty:
            continue
        tf = past.groupby("team")["pace_gap_pct"].mean()
        ss = past.groupby("supplier")["speed_rel"].mean()
        m = d["round"] == rnd
        d.loc[m, "team_form"] = d.loc[m, "team"].map(tf)
        d.loc[m, "supplier_speed"] = d.loc[m, "supplier"].map(ss)
    d["team_form"] = d["team_form"].fillna(d["pace_gap_pct"].mean())
    d["supplier_speed"] = d["supplier_speed"].fillna(0)
    return d


FEATURES = ["quali_gap_pct", "speed_rel", "team_form", "supplier_speed", "grid"]

def models():
    return {
        "1. Qualifying gap only (no ML)":  None,
        "1b. Qualifying gap, rescaled":    Ridge(alpha=1.0),   # same one feature, just scaled
        "2. Linear regression (Ridge)":    Ridge(alpha=1.0),
        "3. Random Forest":                RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42),
        "4. XGBoost":                      XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                        colsample_bytree=0.8, reg_lambda=1.5, random_state=42),
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: WALK FORWARD TEST
# ════════════════════════════════════════════════════════════════════════════
# MAE is how wrong the predicted pace gap is, rank error is how many places out
# the predicted order is.

def evaluate(d):
    d = d.dropna(subset=["pace_gap_pct"])
    results = {name: [] for name in models()}
    results["5. Ensemble (2+3+4 averaged)"] = []
    for rnd in range(6, 15):
        tr, te = d[d["round"] < rnd], d[d["round"] == rnd]
        preds = {}
        for name, mdl in models().items():
            cols = ["quali_gap_pct"] if name.startswith("1b") else FEATURES
            if mdl is None:
                preds[name] = te["quali_gap_pct"].values
            else:
                mdl.fit(tr[cols], tr["pace_gap_pct"])
                preds[name] = mdl.predict(te[cols])
        ens = [k for k in preds if k[0] in "234" and not k.startswith("1b")]
        preds["5. Ensemble (2+3+4 averaged)"] = np.mean([preds[k] for k in ens], axis=0)
        y = te["pace_gap_pct"].values
        for name, p in preds.items():
            mae = np.mean(np.abs(p - y))
            rank_err = np.mean(np.abs(pd.Series(p).rank().values - pd.Series(y).rank().values))
            rho = spearmanr(p, y)[0]
            results[name].append((mae, rank_err, rho))
    rows = []
    for name, v in results.items():
        a = np.array(v)
        rows.append({"model": name, "MAE_%": a[:, 0].mean(), "rank_error_places": a[:, 1].mean(), "rank_corr": a[:, 2].mean()})
    return pd.DataFrame(rows).set_index("model"), results


if __name__ == "__main__":
    d = make_features(pd.read_csv("data/pace_dataset_2026.csv"))
    table, per_round = evaluate(d)
    print("Walk-forward test, rounds 6-14 (train on earlier rounds only)\n")
    print(table.round(3).to_string())
    base = "1. Qualifying gap only (no ML)"
    print("\nMAE by round (baseline vs ensemble):")
    for i, rnd in enumerate(range(6, 15)):
        print(f"  R{rnd:<2} baseline {per_round[base][i][0]:.3f} | ensemble {per_round['5. Ensemble (2+3+4 averaged)'][i][0]:.3f}")
