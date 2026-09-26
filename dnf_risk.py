# dnf_risk.py: works out how likely each car is to retire, and from what.

import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD EVERY RACE RESULT
# ════════════════════════════════════════════════════════════════════════════
# One row per driver per race. Teams are read per round, since lineups change
# mid-season and you can't just map a driver to one team for the whole year.

def load_results():
    rows = []
    races = [(2026, r, f"2026-R{r}") for r in range(1, 15)] + \
            [(y, "Azerbaijan", f"{y}-Baku") for y in range(2021, 2026)]
    for year, rnd, label in races:
        s = fastf1.get_session(year, rnd, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        for r in s.results.itertuples():
            rows.append(dict(race=label, year=year, driver=r.Abbreviation,
                             team=r.TeamName, status=r.Status))
    return pd.DataFrame(rows)


# Which engine each team runs. Engine failures get scored per supplier, since
# four teams sharing a Mercedes are all evidence about the same engine.
SUPPLIER = {
    "Mercedes": "Mercedes", "McLaren": "Mercedes", "Williams": "Mercedes", "Alpine": "Mercedes",
    "Ferrari": "Ferrari", "Haas F1 Team": "Ferrari", "Cadillac": "Ferrari",
    "Red Bull Racing": "Red Bull-Ford", "Racing Bulls": "Red Bull-Ford",
    "Aston Martin": "Honda", "Audi": "Audi",
}


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: ATTACH THE REASON FOR EVERY RETIREMENT
# ════════════════════════════════════════════════════════════════════════════
# FastF1 tells you who retired but almost never why, so the causes come from the
# two CSVs in data/. Categories: power_unit, mechanical, crash, contact,
# tyre_failure, unknown, dns

def build_events(results):
    r26 = pd.read_csv("data/retirements_2026.csv")
    r26["race"] = "2026-R" + r26["round"].astype(str)
    baku = pd.read_csv("data/baku_race_retirements_2021_2025.csv")
    baku["race"] = baku["year"].astype(str) + "-Baku"
    ev = pd.concat([r26[["race", "driver", "category"]], baku[["race", "driver", "category"]]])
    df = results.merge(ev, on=["race", "driver"], how="left")
    df["supplier"] = df["team"].map(SUPPLIER)
    # If you never started you can't fail during the race, so drop those rows
    df = df[df["category"] != "dns"].copy()
    # Retirements with no known cause get counted as mechanical
    df["category"] = df["category"].replace("unknown", "mechanical")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: SHRINKAGE
# ════════════════════════════════════════════════════════════════════════════
# rate = (events + k * grid average) / (starts + k)
# k is basically how many starts worth of doubt to add. Small samples get pulled
# towards the grid average, big ones get trusted. Never gives you 0% or 100%.

def shrunk_rates(df, group_col, causes, k):
    hits = df["category"].isin(causes)
    prior = hits.mean()
    g = df.assign(hit=hits).groupby(group_col)["hit"].agg(["sum", "count"])
    return (g["sum"] + k * prior) / (g["count"] + k)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: ONE DNF PROBABILITY PER CAR
# ════════════════════════════════════════════════════════════════════════════
# Treat the causes as independent, so P(finishes) is just all of them multiplied.
# Engine goes by supplier, mechanical by team, crashes by driver.

def car_probabilities(train, cars, k):
    pu = shrunk_rates(train, "supplier", ["power_unit"], k)
    mech = shrunk_rates(train, "team", ["mechanical"], k)
    inc = shrunk_rates(train, "driver", ["crash", "contact"], k)
    prior = lambda causes: train["category"].isin(causes).mean()
    out = cars.copy()
    out["p_pu"] = out["supplier"].map(pu).fillna(prior(["power_unit"]))
    out["p_mech"] = out["team"].map(mech).fillna(prior(["mechanical"]))
    out["p_incident"] = out["driver"].map(inc).fillna(prior(["crash", "contact"]))
    out["p_dnf"] = 1 - (1 - out.p_pu) * (1 - out.p_mech) * (1 - out.p_incident)
    return out


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: PICK k BY TESTING IT ON PAST RACES
# ════════════════════════════════════════════════════════════════════════════
# Train on earlier rounds only, predict the next one, score with log loss.
# Lower is better, and k=1e9 just means "always use the grid average".

def choose_k(df26, ks):
    rounds = sorted(df26["race"].unique(), key=lambda s: int(s.split("R")[1]))
    scores = {k: [] for k in ks}
    for i in range(5, len(rounds)):
        train, test = df26[df26.race.isin(rounds[:i])], df26[df26.race == rounds[i]]
        actual = (test["category"].notna() & (test["category"] != "tyre_failure")).astype(float).values
        for k in ks:
            p = car_probabilities(train, test, k)["p_dnf"].clip(0.005, 0.95).values
            scores[k].append(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
    return pd.Series({k: np.mean(v) for k, v in scores.items()})


if __name__ == "__main__":
    df = build_events(load_results())
    df26 = df[df.year == 2026]

    ll = choose_k(df26, [1, 5, 10, 20, 30, 50, 100, 1e9])
    print("Log loss by k (lower = better; 1e9 = grid average only)")
    print(ll.round(4).to_string())
    best_k = ll.idxmin()
    print(f"\nBest k = {best_k}")

    # Use the latest lineup, and pool this season with Baku history for crash rates
    lineup = df26[df26.race == "2026-R14"][["driver", "team", "supplier"]]
    probs = car_probabilities(df, lineup, best_k)
    TYRE_FAILURE_BAKU = 2 / 100   # both from 2021, in 100 Baku starts
    probs["p_tyre"] = TYRE_FAILURE_BAKU
    probs["p_dnf"] = 1 - (1 - probs.p_dnf) * (1 - probs.p_tyre)
    print("\nBaku 2026 race-DNF probability per car")
    print((probs.set_index("driver")[["team", "p_pu", "p_mech", "p_incident", "p_dnf"]]
           .sort_values("p_dnf", ascending=False)).round(3).to_string())
    probs.to_csv("data/dnf_probabilities_baku2026.csv", index=False)
