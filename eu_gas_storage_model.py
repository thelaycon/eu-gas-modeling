#!/usr/bin/env python3

import argparse
import io
import json
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from urllib.parse import urlencode

GIE_KEY = "*********************************"
AGSI_URL = "https://agsi.gie.eu/api"
ALSI_URL = "https://alsi.gie.eu/api"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/TTF=F"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
FRED_TTF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

CITIES = [
    ("Berlin", 52.52, 13.41, 0.284),
    ("Paris", 48.85, 2.35, 0.182),
    ("Milan", 45.46, 9.19, 0.193),
    ("Madrid", 40.42, -3.70, 0.136),
    ("Amsterdam", 52.37, 4.90, 0.091),
    ("Warsaw", 52.23, 21.01, 0.114),
]
HDD_BASE = 15.5
CAL_FIRST_WINTER = 2022
WINTER_MONTHS = {10, 11, 12, 1, 2, 3}
CACHE = Path(__file__).resolve().parent / "cache"
TTF_START = date(2023, 1, 1)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (eu-gas-storage-model)"})


class _CurlResponse:
    def __init__(self, text):
        self.text = text

    def json(self):
        return json.loads(self.text)


def curl_get(url, params=None, headers=None, timeout=90):
    full = url
    if params:
        full = f"{url}?{urlencode(params)}"
    cmd = ["curl", "-sS", "-L", "-m", str(timeout), full]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout:
        return _CurlResponse(r.stdout)
    raise RuntimeError(f"curl failed ({r.returncode}): {r.stderr[:200]}")


def http_get(url, params=None, headers=None, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        try:
            return curl_get(url, params, headers)
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}")


def storage_year_of(d):
    return d.year if d.month >= 7 else d.year - 1


def sy_day(d):
    return (d - date(storage_year_of(d), 7, 1)).days


def fetch_gie_raw(base_url, cache_name):
    cache_file = CACHE / cache_name
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 12 * 3600:
        return json.loads(cache_file.read_text())
    rows = []
    page = 1
    last_page = 1
    while page <= last_page:
        payload = None
        for attempt in range(4):
            r = http_get(base_url, params={"type": "eu", "page": page, "size": 100},
                         headers={"x-key": GIE_KEY})
            payload = r.json()
            if payload.get("data"):
                break
            time.sleep(2.0 * (attempt + 1))
        if not payload.get("data"):
            raise RuntimeError(f"empty page {page} from {base_url}")
        last_page = max(last_page, int(payload["last_page"]))
        rows.extend(rec for rec in payload["data"] if rec.get("gasDayStart"))
        page += 1
        time.sleep(0.15)
    CACHE.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(rows))
    return rows


def fetch_agsi():
    rows = fetch_gie_raw(AGSI_URL, "agsi_eu_raw.json")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["gasDayStart"]).dt.normalize()
    for col in ["gasInStorage", "workingGasVolume", "netWithdrawal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.groupby("date")[["gasInStorage", "workingGasVolume",
                             "netWithdrawal"]].mean().sort_index()
    df = df.rename(columns={"gasInStorage": "gis", "workingGasVolume": "wgv",
                            "netWithdrawal": "nwd_field"})
    df["pct"] = df["gis"] / df["wgv"] * 100.0
    return df


def fetch_alsi():
    rows = fetch_gie_raw(ALSI_URL, "alsi_eu_raw.json")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["gasDayStart"]).dt.normalize()
    for col in ["sendOut", "dtrs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["lng_gwh"] = pd.to_numeric(
        df["inventory"].map(lambda x: x.get("gwh")
                            if isinstance(x, dict) else None),
        errors="coerce")
    return df.groupby("date")[["sendOut", "dtrs", "lng_gwh"]].mean().sort_index()


def fetch_weather():
    cache_file = CACHE / "weather_hdd.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["date"]).set_index("date")
    end = date.today() - timedelta(days=6)
    frames = []
    for name, lat, lon, weight in CITIES:
        chunks = [(date(2016, 1, 1), date(2020, 12, 31)),
                  (date(2021, 1, 1), end)]
        temps = {}
        for start, stop in chunks:
            params = {"latitude": lat, "longitude": lon,
                      "start_date": start.isoformat(), "end_date": stop.isoformat(),
                      "daily": "temperature_2m_mean",
                      "timezone": "Europe/Berlin"}
            r = http_get(OPEN_METEO_URL, params=params)
            daily = r.json()["daily"]
            for ds, t in zip(daily["time"], daily["temperature_2m_mean"]):
                temps[ds] = None if t is None else t
        s = pd.Series(temps, name=name, dtype="float64")
        s.index = pd.to_datetime(s.index)
        frames.append(s)
        time.sleep(0.3)
    temps = pd.concat(frames, axis=1)
    hdd_cities = temps.clip(lower=temps - HDD_BASE, upper=None)
    hdd_cities = (HDD_BASE - temps).clip(lower=0)
    weights = np.array([w for _, _, _, w in CITIES])
    weights = weights / weights.sum()
    hdd = hdd_cities.fillna(0.0).values @ weights
    out = pd.DataFrame({"hdd": hdd}, index=temps.index)
    out.index.name = "date"
    CACHE.mkdir(exist_ok=True)
    out.to_csv(cache_file)
    return out


def fetch_ttf():
    cache_file = CACHE / "ttf_fred_monthly.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["date"]).set_index("date")
    r = http_get(FRED_TTF_URL, params={"id": "PNGASEUUSDM"})
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"observation_date": "date", "PNGASEUUSDM": "ttf"})
    df["date"] = pd.to_datetime(df["date"])
    df["ttf"] = pd.to_numeric(df["ttf"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    df = df[df.index >= pd.Timestamp(2016, 1, 1)]
    CACHE.mkdir(exist_ok=True)
    df.to_csv(cache_file)
    return df


def fetch_yahoo_latest():
    cache_file = CACHE / "ttf_yahoo_latest.json"
    try:
        r = http_get(YAHOO_URL, params={"interval": "1d", "range": "5d"})
        res = r.json()["chart"]["result"][0]
        meta = res["meta"]
        price = float(meta["regularMarketPrice"])
        ts = meta.get("regularMarketTime")
        d = datetime.fromtimestamp(ts).date() if ts else date.today()
        cache_file.parent.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps({"date": d.isoformat(), "ttf": price}))
    except Exception:
        if not cache_file.exists():
            raise
    j = json.loads(cache_file.read_text())
    return j["ttf"], date.fromisoformat(j["date"])


def five_year_deviation(agsi):
    df = agsi.copy()
    df["sy"] = [storage_year_of(d) for d in df.index.date]
    df["syd"] = [sy_day(d) for d in df.index.date]
    pivot = df.pivot_table(index="syd", columns="sy", values="pct")
    records = []
    for idx, row in df.iterrows():
        d = idx.date()
        sy, syd = int(row["sy"]), int(row["syd"])
        prev = [pivot.get(c) for c in range(sy - 5, sy) if c in pivot.columns]
        if len(prev) < 5:
            records.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        vals = [col.loc[k] for col in prev for k in range(syd - 3, syd + 4)
                if k in col.index]
        window = pd.Series(vals).dropna()
        if window.empty:
            records.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        cur = row["pct"]
        records.append((idx, window.mean(), window.min(), window.max(),
                        cur - window.mean(),
                        (cur - window.min()) / (window.max() - window.min()) * 100))
    out = pd.DataFrame(records, columns=["date", "mean5y", "min5y", "max5y",
                                         "dev_pp", "position_pct"]).set_index("date")
    return out


def fit_model(agsi, weather, exclude_winter=None):
    df = agsi.join(weather, how="inner").copy()
    df["nwd"] = df["nwd_field"]
    gap = (df["gis"].diff(-1) * 1000.0)
    df["nwd"] = df["nwd"].fillna(gap)
    df = df.dropna(subset=["nwd", "hdd"])
    df["sy"] = [storage_year_of(d) for d in df.index.date]
    df["month"] = df.index.month
    df = df[df["month"].isin(WINTER_MONTHS) & (df["sy"] >= CAL_FIRST_WINTER)]
    if exclude_winter is not None:
        df = df[df["sy"] != exclude_winter]
    months = sorted(WINTER_MONTHS)
    X = np.column_stack([df["hdd"].values] +
                        [(df["month"].values == m).astype(float) for m in months])
    X = sm.add_constant(X, prepend=False)
    res = sm.OLS(df["nwd"].values, X).fit()
    nhd = len(months)
    beta = res.params[0]
    month_intercepts = {m: res.params[i + 1] for i, m in enumerate(months)}
    return {"beta": beta, "months": months, "intercepts": month_intercepts,
            "r2": res.rsquared, "n": int(res.nobs), "t_beta": res.tvalues[0],
            "resid_std": float(np.sqrt(res.mse_resid)), "params": res.params,
            "design": X}


def predict_nwd(model, month, hdd):
    months = model["months"]
    x = np.array([hdd] + [1.0 if month == m else 0.0 for m in months] + [1.0])
    return float(x @ model["params"])


def simulate(start_twh, wgv, sim_dates, hdd_by_date, model):
    s = start_twh
    for d in sim_dates:
        h = hdd_by_date.get(d)
        if h is None:
            h = hdd_by_date.get((d.month, d.day))
        if h is None or np.isnan(h):
            continue
        s -= predict_nwd(model, d.month, h) / 1000.0
        s = min(max(s, 0.0), wgv)
    return s


def asof_pct(df, target):
    sub = df.loc[df.index.date <= target, "pct"]
    return None if sub.empty else float(sub.iloc[-1]), sub.index[-1].date()


def main():
    ap = argparse.ArgumentParser(description="EU gas storage model")
    ap.add_argument("--target", type=float, default=30.0,
                    help="1 April target, %% of working gas (default 30)")
    ap.add_argument("--backtest-date", type=str, default="02-01",
                    help="MM-DD projection date for backtest (default 02-01)")
    ap.add_argument("--cargo-twh", type=float, default=1.2,
                    help="TWh per standard LNG cargo (default 1.2)")
    args = ap.parse_args()

    agsi = fetch_agsi()
    alsi = fetch_alsi()
    weather = fetch_weather()
    ttf = fetch_ttf()

    last_day = agsi.index[-1].date()
    first_day = agsi.index[0].date()
    gis = float(agsi["gis"].iloc[-1])
    wgv = float(agsi["wgv"].iloc[-1])
    pct = float(agsi["pct"].iloc[-1])

    print("=" * 74)
    print("EU GAS STORAGE MODEL")
    print("=" * 74)
    print(f"AGSI data: {first_day} to {last_day} (gas day)")
    print(f"Storage: {gis:.1f} TWh in {wgv:.1f} TWh working gas = {pct:.1f}% full")
    recent = agsi["nwd_field"].dropna().iloc[-7:]
    print(f"Net withdrawals, last 7 days: {recent.mean():.0f} GWh/d average")

    band = five_year_deviation(agsi)
    cur = band.iloc[-1]
    print()
    print("-" * 74)
    print(f"5-YEAR BAND (storage years from 1 July; comparing {last_day})")
    print("-" * 74)
    print(f"Current % full                 : {pct:.1f}")
    print(f"5-year mean for this day       : {cur['mean5y']:.1f}")
    print(f"5-year range for this day      : {cur['min5y']:.1f} - {cur['max5y']:.1f}")
    print(f"Deviation vs 5-year mean       : {cur['dev_pp']:+.1f} pp")
    pos_txt = (f"{cur['position_pct']:.0f}% of range"
               if cur['position_pct'] >= 0 else "below the 5-year low")
    print(f"Position in 5-year range       : {pos_txt} (0=lowest, 100=highest)")

    model = fit_model(agsi, weather)
    print()
    print("-" * 74)
    print("WEATHER MODEL: daily net withdrawals ~ HDD + month intercepts")
    print("-" * 74)
    print(f"Calibration: winters 2022/23 onward, Oct-Mar days (n={model['n']}), "
          f"R2={model['r2']:.3f}")
    print(f"HDD coefficient: {model['beta']:.1f} GWh/d per HDD "
          f"(t={model['t_beta']:.1f}); residual sd {model['resid_std']:.0f} GWh/d")
    ints = ", ".join(f"{m:02d}:{model['intercepts'][m]:+.0f}"
                     for m in model["months"])
    print(f"Month intercepts (GWh/d): {ints}")

    sim_start = last_day + timedelta(days=1)
    sim_end = date(storage_year_of(sim_start) + 1, 3, 31)
    sim_dates = []
    d = sim_start
    while d <= sim_end:
        sim_dates.append(d)
        d += timedelta(days=1)

    weather_dates = weather.index.date
    weather_by_date = dict(zip(weather_dates, weather["hdd"].values))
    winters_weather = {}
    for y in range(2016, storage_year_of(last_day)):
        wm = {}
        start = date(y, 10, 1)
        end = date(y + 1, 3, 31)
        dd = start
        while dd <= end:
            if dd in weather_by_date:
                wm[(dd.month, dd.day)] = weather_by_date[dd]
            dd += timedelta(days=1)
        if len(wm) > 120:
            winters_weather[y] = wm

    scenarios = []
    for y, wm in sorted(winters_weather.items()):
        proj = simulate(gis, wgv, sim_dates, wm, model)
        scenarios.append((y, proj / wgv * 100, proj))
    scen = pd.DataFrame(scenarios, columns=["winter", "apr1_pct", "apr1_twh"])

    print()
    print("-" * 74)
    print(f"WINTER SCENARIOS: rest of winter ({sim_start} to {sim_end}) rerun under "
          f"each past winter's weather")
    print("-" * 74)
    for _, r in scen.iterrows():
        y = int(r["winter"])
        print(f"  Winter {y}/{str(y + 1)[-2:]}: "
              f"1 April {r['apr1_pct']:.1f}% full ({r['apr1_twh']:.0f} TWh)")
    cold = float(np.percentile(scen["apr1_pct"], 10))
    normal = float(np.percentile(scen["apr1_pct"], 50))
    mild = float(np.percentile(scen["apr1_pct"], 90))
    print(f"  COLD  (10th pct): {cold:.1f}% full on 1 April")
    print(f"  NORMAL (median) : {normal:.1f}% full on 1 April")
    print(f"  MILD  (90th pct): {mild:.1f}% full on 1 April")

    mm, dd_ = (int(x) for x in args.backtest_date.split("-"))
    bt_results = []
    print()
    print("-" * 74)
    print(f"BACKTEST: 1 April projection made on {mm:02d}-{dd_:02d} "
          f"(leave-one-winter-out)")
    print("-" * 74)
    for y in range(2017, storage_year_of(last_day)):
        start = date(y + 1, mm, dd_)
        actual_pair = asof_pct(agsi, date(y + 1, 4, 1))
        start_pair = asof_pct(agsi, start)
        if actual_pair is None or start_pair is None:
            continue
        actual_pct, actual_day = actual_pair
        start_pct, start_day = start_pair
        wgv_y = float(agsi.loc[(agsi.index.date >= date(y, 7, 1)) &
                               (agsi.index.date <= date(y + 1, 6, 30)),
                               "wgv"].median())
        cal_w = y if y >= CAL_FIRST_WINTER else None
        m_bt = fit_model(agsi, weather, exclude_winter=cal_w)
        sim_d = []
        dd2 = date(y + 1, mm, dd_)
        end2 = date(y + 1, 3, 31)
        while dd2 <= end2:
            sim_d.append(dd2)
            dd2 += timedelta(days=1)
        hdd_map = {x: weather_by_date.get(x) for x in sim_d}
        proj = simulate(start_pct / 100 * wgv_y, wgv_y, sim_d, hdd_map, m_bt)
        pred_pct = proj / wgv_y * 100
        err = pred_pct - actual_pct
        bt_results.append((y, start_day, pred_pct, actual_pct, err))
    bt = pd.DataFrame(bt_results, columns=["winter", "proj_from", "pred_pct",
                                           "actual_pct", "err_pp"])
    for _, r in bt.iterrows():
        print(f"  Winter {r['winter']}/{str(r['winter'] + 1)[-2:]}: predicted "
              f"{r['pred_pct']:.1f}% vs actual {r['actual_pct']:.1f}% "
              f"-> error {r['err_pp']:+.1f} pp")
    print(f"  Mean absolute error: {bt['err_pp'].abs().mean():.1f} pp "
          f"(mean bias {bt['err_pp'].mean():+.1f} pp)")

    print()
    print("-" * 74)
    print(f"LNG NEEDS vs target of {args.target:.0f}% full on 1 April")
    print("-" * 74)
    target_twh = args.target / 100 * wgv
    days_to_apr = (date(storage_year_of(last_day) + 1, 4, 1) - last_day).days
    send_out = float(alsi["sendOut"].dropna().iloc[-1])
    regas_cap = float(alsi["dtrs"].dropna().iloc[-1])
    spare_regas = regas_cap - send_out
    lng_inv = float(alsi["lng_gwh"].dropna().iloc[-1]) / 1000.0
    print(f"Target level: {target_twh:.0f} TWh; days to 1 April: {days_to_apr}")
    print(f"Current LNG inventory (ALSI): {lng_inv:.1f} TWh; send-out "
          f"{send_out:.0f} GWh/d of {regas_cap:.0f} GWh/d capacity "
          f"(spare {spare_regas:.0f} GWh/d)")
    for label, val in [("COLD", cold), ("NORMAL", normal), ("MILD", mild)]:
        level = val / 100 * wgv
        gap = level - target_twh
        cargoes = gap / args.cargo_twh
        extra = abs(gap) * 1000 / days_to_apr
        share = extra / spare_regas * 100
        direction = "surplus" if gap > 0 else "shortfall"
        print(f"  {label:6s}: {val:.1f}% ({level:.0f} TWh) -> {direction} "
              f"{abs(gap):.1f} TWh vs target")
        if gap > 0:
            print(f"          = {cargoes:.1f} cargoes surplus; extra send-out "
                  f"{extra:.0f} GWh/d = {share:.1f}% of spare regas capacity")
        else:
            print(f"          = {abs(cargoes):.1f} cargoes needed; send-out cut "
                  f"of {extra:.0f} GWh/d (or equal extra injection) required "
                  f"= {share:.1f}% of spare regas capacity")

    dev = band[["dev_pp"]].copy()
    dev_m = dev.resample("MS").mean().dropna()
    merged = dev_m.join(ttf, how="inner").dropna()
    merged = merged[merged.index >= pd.Timestamp(TTF_START)]
    ttf_yahoo, ttf_yahoo_day = fetch_yahoo_latest()
    fred_last = ttf.iloc[-1]
    print()
    print("-" * 74)
    print(f"TTF LINK (descriptive, since {TTF_START}; monthly, n={len(merged)} months)")
    print("-" * 74)
    print(f"Latest Yahoo TTF front-month: {ttf_yahoo:.2f} EUR/MWh ({ttf_yahoo_day})")
    print(f"Latest FRED IMF EU gas price (TTF proxy): {fred_last['ttf']:.2f} "
          f"USD/MMBtu ({ttf.index[-1].date()})")
    r_level = float(np.corrcoef(merged["ttf"], merged["dev_pp"])[0, 1])
    chg = merged.diff(3).dropna()
    r_chg = float(np.corrcoef(chg["ttf"], chg["dev_pp"])[0, 1])
    print(f"Correlation price level vs storage deviation vs 5y mean: "
          f"r = {r_level:+.2f}")
    print(f"Correlation 3-month change in price vs change in deviation: "
          f"r = {r_chg:+.2f}")
    print("  (negative means low storage vs 5y average goes with higher price)")


if __name__ == "__main__":
    main()
