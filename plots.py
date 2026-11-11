#!/usr/bin/env python3
"""Charts for the EU gas storage model. Run after eu_gas_storage_model.py."""
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import eu_gas_storage_model as m

OUT = Path(__file__).resolve().parent / "charts"
OUT.mkdir(exist_ok=True)

CREAM = "#f7f3ea"
DARK = "#20201c"
BLUE = "#1f4e79"
BAND = "#c7d1dc"
GRAY = "#a49c8c"
RED = "#b0432f"
GREEN = "#3f6d54"
ORANGE = "#c97b2d"
GRID = "#e6dfd0"
SERIF = "Liberation Serif"

plt.rcParams.update({
    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM,
    "axes.edgecolor": "#8a8577",
    "axes.labelcolor": DARK,
    "axes.titlecolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "font.family": SERIF,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

MONTH_TICKS = [(7, "Jul"), (8, "Aug"), (9, "Sep"), (10, "Oct"), (11, "Nov"),
               (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr"),
               (5, "May"), (6, "Jun")]


def month_ticks(ax, year):
    pos, lab = [], []
    for mo, name in MONTH_TICKS:
        d = date(year, mo, 1)
        pos.append(m.sy_day(d))
        lab.append(name)
    ax.set_xticks(pos)
    ax.set_xticklabels(lab)
    ax.set_xlim(0, 366)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main():
    agsi = m.fetch_agsi()
    alsi = m.fetch_alsi()
    weather = m.fetch_weather()
    ttf = m.fetch_ttf()
    band = m.five_year_deviation(agsi)
    model = m.fit_model(agsi, weather)
    last_day = agsi.index[-1].date()
    gis = float(agsi["gis"].iloc[-1])
    wgv = float(agsi["wgv"].iloc[-1])
    pct = float(agsi["pct"].iloc[-1])
    cur_sy = m.storage_year_of(last_day)

    df = agsi.copy()
    df["sy"] = [m.storage_year_of(d) for d in df.index.date]
    df["syd"] = [m.sy_day(d) for d in df.index.date]
    pivot = df.pivot_table(index="syd", columns="sy", values="pct")
    prev_years = list(range(cur_sy - 5, cur_sy))

    # 1 hero: storage vs 5-year band -------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.6))
    lo = pivot[prev_years].min(axis=1)
    hi = pivot[prev_years].max(axis=1)
    mean5 = pivot[prev_years].mean(axis=1)
    ax.fill_between(lo.index, lo.values, hi.values, color=BAND, alpha=0.55,
                    label="5-year range", zorder=1)
    ax.plot(mean5.index, mean5.values, color=GRAY, lw=1.6, ls="--",
            label="5-year mean", zorder=2)
    for y in prev_years:
        ax.plot(pivot.index, pivot[y], color=GRAY, lw=0.8, alpha=0.45, zorder=2)
    ax.plot(pivot.index, pivot[cur_sy], color=BLUE, lw=2.6, zorder=4,
            label=f"{cur_sy}/{str(cur_sy + 1)[-2:]} season")
    ax.plot([m.sy_day(last_day)], [pct], "o", color=BLUE, ms=9, zorder=5,
            mec="white", mew=1.4)
    ax.annotate(f"today: {pct:.1f}%\n({pct - mean5.loc[m.sy_day(last_day)]:+.1f} pp "
                f"vs 5-yr mean)", xy=(m.sy_day(last_day), pct),
                xytext=(-118, -46), textcoords="offset points",
                fontsize=11, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of working gas capacity")
    ax.set_title(f"EU underground gas storage vs the 5-year band "
                 f"(to {last_day.isoformat()})")
    month_ticks(ax, cur_sy)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    save(fig, "01_storage_vs_band.png")

    # 2 deviation area ----------------------------------------------------
    cur = df[df["sy"] == cur_sy].join(band["dev_pp"], rsuffix="_b")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = pd.to_datetime(cur.index)
    vals = cur["dev_pp"].values
    ax.axhline(0, color="#8a8577", lw=1)
    ax.fill_between(x, vals, 0, where=(vals < 0), color=RED, alpha=0.75,
                    interpolate=True)
    ax.fill_between(x, vals, 0, where=(vals >= 0), color=GREEN, alpha=0.75,
                    interpolate=True)
    ax.plot(x, vals, color=DARK, lw=1.8)
    ax.annotate(f"{vals[-1]:+.1f} pp", xy=(x[-1], vals[-1]),
                xytext=(-14, 14), textcoords="offset points",
                fontsize=12, fontweight="bold",
                color=RED if vals[-1] < 0 else GREEN,
                arrowprops=dict(arrowstyle="-", color=RED if vals[-1] < 0
                                else GREEN, lw=1))
    ax.set_ylabel("deviation vs 5-yr mean, pp")
    ax.set_title(f"Storage deviation from the 5-year average, "
                 f"{cur_sy}/{str(cur_sy + 1)[-2:]} season")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(axis="y", color=GRID, lw=0.7)
    save(fig, "02_deviation.png")

    # 3 scenario fan -------------------------------------------------------
    sim_start = last_day + timedelta(days=1)
    sim_end = date(cur_sy + 1, 3, 31)
    sim_dates = []
    d = sim_start
    while d <= sim_end:
        sim_dates.append(d)
        d += timedelta(days=1)
    weather_by_date = dict(zip(weather.index.date, weather["hdd"].values))
    winters_weather = {}
    for y in range(2016, cur_sy):
        wm, dd = {}, date(y, 10, 1)
        while dd <= date(y + 1, 3, 31):
            if dd in weather_by_date:
                wm[(dd.month, dd.day)] = weather_by_date[dd]
            dd += timedelta(days=1)
        if len(wm) > 120:
            winters_weather[y] = wm
    paths = {}
    for y, wm in sorted(winters_weather.items()):
        s = gis
        pp = []
        for d in sim_dates:
            h = wm.get((d.month, d.day))
            if h is not None and not np.isnan(h):
                s -= m.predict_nwd(model, d.month, h) / 1000.0
                s = min(max(s, 0.0), wgv)
            pp.append(s / wgv * 100)
        paths[y] = pp
    arr = np.array([paths[y] for y in sorted(paths)])
    xd = mdates.date2num(pd.to_datetime(sim_dates))
    target = 30.0
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.fill_between(xd, np.percentile(arr, 10, axis=0),
                    np.percentile(arr, 90, axis=0), color=BAND, alpha=0.6,
                    label="10th-90th pct of past winters")
    for y in sorted(paths):
        ax.plot(xd, paths[y], color=GRAY, lw=0.7, alpha=0.5)
    med = np.percentile(arr, 50, axis=0)
    ax.plot(xd, med, color=BLUE, lw=2.4, label="median of past winters")
    ax.axhline(target, color=ORANGE, lw=1.8, ls="--")
    ax.annotate(f"target {target:.0f}%", xy=(xd[3], target), fontsize=10,
                color=ORANGE, fontweight="bold", va="bottom")
    ax.plot([mdates.date2num(pd.Timestamp(last_day))], [pct], "o", color=BLUE,
            ms=8, mec="white", mew=1.3, zorder=5)
    for qp, lab, col in [(10, "cold", RED), (50, "normal", BLUE),
                         (90, "mild", GREEN)]:
        v = float(np.percentile(arr[:, -1], qp))
        ax.plot([xd[-1]], [v], "o", color=col, ms=8, mec="white", mew=1.3,
                zorder=5)
        ax.annotate(f"{lab}: {v:.0f}%", xy=(xd[-1], v), xytext=(7, 0),
                    textcoords="offset points", color=col, fontsize=10.5,
                    fontweight="bold", va="center")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylabel("% of working gas capacity")
    ax.set_title("Projected EU storage to 1 April under each past winter's weather")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_ylim(0, 80)
    save(fig, "03_scenarios.png")

    # 4 backtest dumbbells --------------------------------------------------
    mm, dd_ = 2, 1
    rows = []
    for y in range(2017, cur_sy):
        ap = m.asof_pct(agsi, date(y + 1, 4, 1))
        sp = m.asof_pct(agsi, date(y + 1, mm, dd_))
        if ap is None or sp is None:
            continue
        actual_pct, _ = ap
        start_pct, _ = sp
        wgv_y = float(agsi.loc[(agsi.index.date >= date(y, 7, 1)) &
                               (agsi.index.date <= date(y + 1, 6, 30)),
                               "wgv"].median())
        m_bt = m.fit_model(agsi, weather,
                           exclude_winter=y if y >= m.CAL_FIRST_WINTER else None)
        sim_d = []
        dd2 = date(y + 1, mm, dd_)
        while dd2 <= date(y + 1, 3, 31):
            sim_d.append(dd2)
            dd2 += timedelta(days=1)
        hmap = {x: weather_by_date.get(x) for x in sim_d}
        proj = m.simulate(start_pct / 100 * wgv_y, wgv_y, sim_d, hmap, m_bt)
        rows.append((y, proj / wgv_y * 100, actual_pct))
    bt = pd.DataFrame(rows, columns=["y", "pred", "actual"])
    bt["err"] = bt["pred"] - bt["actual"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    yy = np.arange(len(bt))
    for i, r in bt.reset_index(drop=True).iterrows():
        ax.plot([r["actual"], r["pred"]], [i, i], color="#d5cfc0", lw=2.4,
                zorder=1)
    ax.scatter(bt["actual"], yy, s=58, color=DARK, zorder=3,
               label="actual 1 April")
    ax.scatter(bt["pred"], yy, s=58, color=ORANGE, zorder=3,
               label="model projection (from 1 Feb)")
    for i, r in bt.reset_index(drop=True).iterrows():
        ax.annotate(f"{r['err']:+.1f}", xy=(max(r["pred"], r["actual"]), i),
                    xytext=(10, -3), textcoords="offset points", fontsize=9,
                    color=RED if abs(r["err"]) >= 5 else "#555555")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{int(y)}/{str(y + 1)[-2:]}" for y in bt["y"]])
    ax.invert_yaxis()
    ax.set_xlabel("% of working gas capacity on 1 April")
    ax.set_title("Backtest: 1 April projection vs what actually happened")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "04_backtest.png")

    # 5 weather model scatter ------------------------------------------------
    reg = agsi.join(weather, how="inner").dropna(subset=["nwd_field", "hdd"])
    reg = reg[reg.index.month.isin(m.WINTER_MONTHS)]
    reg = reg[[m.storage_year_of(d) >= m.CAL_FIRST_WINTER
               for d in reg.index.date]]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    cmap = {10: "#e67e22", 11: "#c0392b", 12: "#8e44ad", 1: "#0b3d91",
            2: "#1e8449", 3: "#16a085"}
    for mo in sorted(m.WINTER_MONTHS):
        sub = reg[reg.index.month == mo]
        ax.scatter(sub["hdd"], sub["nwd_field"], s=14, alpha=0.5,
                   color=cmap[mo],
                   label={10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan",
                          2: "Feb", 3: "Mar"}[mo])
    xs = np.linspace(0, reg["hdd"].max(), 100)
    ax.plot(xs, model["beta"] * xs + model["intercepts"][1], color=DARK,
            lw=2.2, label="fit path shown for January")
    ax.annotate(f"slope: {model['beta']:.0f} GWh/d per HDD\n"
                f"R2 = {model['r2']:.2f}  (winter days 2022/23 onward)",
                xy=(0.03, 0.95), xycoords="axes fraction", fontsize=11,
                va="top")
    ax.set_xlabel("demand-weighted heating degree days (6 countries)")
    ax.set_ylabel("net withdrawals, GWh/d")
    ax.set_title("Cold days pull gas out of storage, fast", pad=36)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, ncol=7, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=9.5, columnspacing=0.9)
    save(fig, "05_weather_model.png")

    # 6 LNG infographic ------------------------------------------------------
    target_pct = 30.0
    target_twh = target_pct / 100 * wgv
    cargo = 1.2
    scen_pcts = {lab: float(np.percentile(arr[:, -1], qp))
                 for lab, qp in [("Cold", 10), ("Normal", 50), ("Mild", 90)]}
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    ax.set_axis_off()
    ax.set_xlim(-2.8, 20.8)
    ax.set_ylim(-0.55, 3.45)

    def draw_ship(x0, y0, s=1.0, frac=1.0):
        ax.add_patch(plt.Polygon(
            [[x0, y0], [x0 + 0.95 * s, y0], [x0 + 0.80 * s, y0 + 0.22 * s],
             [x0 + 0.12 * s, y0 + 0.22 * s]],
            closed=True, facecolor="#33322c", zorder=3))
        for i in range(4):
            ax.add_patch(plt.Circle((x0 + (0.20 + i * 0.185) * s,
                                     y0 + 0.34 * s), 0.098 * s,
                                    facecolor=RED, edgecolor="#33322c",
                                    lw=0.7, zorder=3))
        ax.add_patch(plt.Rectangle((x0 + 0.80 * s, y0 + 0.22 * s),
                                   0.09 * s, 0.17 * s,
                                   facecolor="#33322c", zorder=3))
        if frac < 1.0:
            ax.add_patch(plt.Rectangle((x0 + frac * 1.02 * s, y0 - 0.04 * s),
                                       (1.06 - frac) * s, 0.68 * s,
                                       facecolor=CREAM, zorder=4))

    ys = {"Cold": 2.4, "Normal": 1.2, "Mild": 0.0}
    for lab, p in scen_pcts.items():
        gap = p / 100 * wgv - target_twh
        cargoes = abs(gap) / cargo
        y = ys[lab]
        ax.text(-0.35, y + 0.18, lab, ha="right", va="center", fontsize=13,
                fontweight="bold")
        n_full = int(cargoes // 10)
        rem = (cargoes - n_full * 10) / 10
        x = 0.0
        for _ in range(n_full):
            draw_ship(x, y)
            x += 1.06
        if rem > 0.03:
            draw_ship(x, y, frac=rem)
            x += 1.06
        ax.text(x + 0.2, y + 0.18, f"{abs(gap):.0f} TWh short "
                f"— {cargoes:.0f} cargoes", va="center", fontsize=10.5,
                color="#33322c")
    ax.text(0.0, 3.25, "1 ship = 10 standard cargoes (~12 TWh each)",
            fontsize=9.5, color="#6b665a")
    ax.set_title("How much LNG stands between the scenarios and the 30% target",
                 pad=14)
    save(fig, "06_lng_gaps.png")

    # 7 ttf link -------------------------------------------------------------
    dev_m = band[["dev_pp"]].resample("MS").mean().dropna()
    merged = dev_m.join(ttf, how="inner").dropna()
    merged = merged[merged.index >= pd.Timestamp(m.TTF_START)]
    r = float(np.corrcoef(merged["ttf"], merged["dev_pp"])[0, 1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(merged.index, merged["dev_pp"], width=20, color=RED, alpha=0.65,
           label="storage deviation vs 5-yr mean (left)")
    ax.axhline(0, color="#8a8577", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(merged.index, merged["ttf"], color=DARK, lw=2.2, marker="o",
             ms=3.5, label="EU gas price, IMF monthly (right)")
    ax.set_ylabel("storage deviation, pp")
    ax2.set_ylabel("USD/MMBtu")
    ax2.spines["right"].set_visible(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_title(f"Price and storage move together (r = {r:+.2f}, monthly since 2023)",
                 pad=36)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, ncol=2, frameon=False,
              fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax.grid(axis="y", color=GRID, lw=0.7)
    save(fig, "07_ttf_link.png")


if __name__ == "__main__":
    main()
