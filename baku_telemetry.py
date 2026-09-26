# baku_telemetry.py: speed traces around the lap, with the corners marked.

import warnings
import numpy as np
import matplotlib.pyplot as plt
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

# Turn 3 and Turn 15 cause most of the crashes here, Turn 4 a distant third.
RISK_CORNERS = {3, 4, 15}


def corner_distances(reference_lap):
    """Matches each corner's coordinates to the nearest point on a real lap, so the
    markers line up with the same distance axis as the speed trace."""
    tel = reference_lap.get_telemetry()
    xy = tel[["X", "Y"]].values
    corners = reference_lap.session.get_circuit_info().corners
    out = {}
    for row in corners.itertuples():
        d = np.hypot(xy[:, 0] - row.X, xy[:, 1] - row.Y)
        out[int(row.Number)] = tel["Distance"].values[np.argmin(d)]
    return out


def speed_trace(drivers=("VER", "RUS", "LEC")):
    q = fastf1.get_session(2025, "Azerbaijan", "Q")
    q.load(telemetry=True, weather=False, messages=False)
    ref_lap = q.laps.pick_drivers(drivers[0]).pick_fastest()
    corners = corner_distances(ref_lap)

    fig, ax = plt.subplots(figsize=(13, 6))
    for drv in drivers:
        lap = q.laps.pick_drivers(drv).pick_fastest()
        tel = lap.get_telemetry()
        ax.plot(tel["Distance"], tel["Speed"], label=f"{drv} ({lap['LapTime']})", linewidth=1.6)
    for num, dist in sorted(corners.items()):
        is_risk = num in RISK_CORNERS
        ax.axvline(dist, color="#c00000" if is_risk else "grey",
                   linestyle="-" if is_risk else ":", alpha=0.8 if is_risk else 0.5)
        ax.text(dist, 60, f"T{num}", rotation=90, fontsize=7, ha="right", va="bottom",
                color="#c00000" if is_risk else "dimgrey")
    ax.set_xlabel("Distance around the lap (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Baku - real speed trace, 2025 qualifying (fastest laps)\n"
                 "Red lines = Turns 3, 4, 15 - the corners with the most crashes, 2021-2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig("data/baku_speed_trace_2025.png", dpi=150)
    plt.close(fig)
    top_speed = {drv: q.laps.pick_drivers(drv).pick_fastest().get_car_data()["Speed"].max() for drv in drivers}
    return top_speed


def deployment_compare(year=2026, rnd=15, a="RUS", b="LEC"):
    """Compares two qualifying laps to show where each driver spends their battery.
    In 2026 roughly half the power is electric, so a driver can deliberately give up
    speed in one part of the lap to have more left for the straight."""
    q = fastf1.get_session(year, rnd, "Q")
    q.load(telemetry=True, weather=False, messages=False)
    laps = {d: q.laps.pick_drivers(d).pick_fastest() for d in (a, b)}
    tel = {d: laps[d].get_telemetry() for d in (a, b)}
    grid = np.arange(0, int(min(t["Distance"].max() for t in tel.values())), 10)
    sp = {d: np.interp(grid, tel[d]["Distance"], tel[d]["Speed"]) for d in (a, b)}
    delta = sp[a] - sp[b]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for d, c in [(a, "#00b3a4"), (b, "#c00000")]:
        ax1.plot(grid, sp[d], label=f"{d} ({laps[d]['LapTime']})", color=c, linewidth=1.6)
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title(f"Where the lap time comes from - {a} vs {b}, Baku qualifying")
    ax1.legend(loc="lower right")

    ax2.fill_between(grid, delta, 0, where=delta >= 0, color="#00b3a4", alpha=0.6)
    ax2.fill_between(grid, delta, 0, where=delta < 0, color="#c00000", alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel(f"{a} faster (+) / slower (-)\nkm/h")
    ax2.set_xlabel("Distance around the lap (m)")
    fig.tight_layout()
    fig.savefig("data/baku_deployment_compare.png", dpi=150)
    plt.close(fig)

    early = (grid < 500)
    straight = (grid > 4500)
    return {"worst_early_deficit": delta[early].min(),
            "straight_advantage": delta[straight].mean(),
            f"{a}_top_speed": sp[a].max(), f"{b}_top_speed": sp[b].max()}


if __name__ == "__main__":
    d = deployment_compare()
    print("Russell's energy deployment trade:")
    print(f"  slower by up to {abs(d['worst_early_deficit']):.0f} km/h in the opening corners")
    print(f"  then {d['straight_advantage']:+.1f} km/h on average down the final straight")
    print(f"  top speed {d['RUS_top_speed']:.0f} vs {d['LEC_top_speed']:.0f} km/h")
    print("Saved: data/baku_deployment_compare.png\n")

    top_speed = speed_trace()
    print("Top speed reached on the main straight, Baku 2025 qualifying:")
    for d, v in sorted(top_speed.items(), key=lambda x: -x[1]):
        print(f"  {d}: {v:.0f} km/h")
    print("\nSaved: data/baku_speed_trace_2025.png")
