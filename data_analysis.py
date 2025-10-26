# eda_pollution.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RAW_CSV = "data.csv"         
CLEAN_CSV = "data_clean.csv"    
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 4),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False
})

def parse_dt(x: str) -> datetime:
    """Parse 'year month day hour' into datetime."""
    return datetime.strptime(x, "%Y %m %d %H")

WIND_DEG = {
    "N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,"SE":135,"SSE":157.5,
    "S":180,"SSW":202.5,"SW":225,"WSW":247.5,"W":270,"WNW":292.5,"NW":315,"NNW":337.5
}


def load_and_clean(raw_csv=RAW_CSV, out_csv=CLEAN_CSV) -> pd.DataFrame:
    # Columns (original & new)
    orig_cols = ["No","year","month","day","hour","pm2.5","DEWP","TEMP","PRES","cbwd","Iws","Is","Ir"]
    new_cols  = ["pollution","dew","temp","pressure","w_dir","w_speed","snow","rain"]

    df = pd.read_csv(
        raw_csv,
        parse_dates=[["year","month","day","hour"]],
        date_parser=parse_dt,
        index_col=0
    )

    if "No" in df.columns:
        df = df.drop(columns=["No"])

    df.columns = new_cols
    df = df.sort_index()

    num_like = ["pollution","dew","temp","pressure","w_speed","snow","rain"]
    for c in num_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # wind direction to tidy categorical
    df["w_dir"] = df["w_dir"].astype(str).str.upper().str.strip()

    # handle pollution NAs (forward fill -> back fill -> 0 as last resort)
    df["pollution"] = df["pollution"].clip(lower=0)  # no negatives
    df["pollution"] = df["pollution"].ffill().bfill().fillna(0)

    # drop the first day (same effect as dataset[24:] but robust to missing hours)
    if len(df.index) > 0:
        first_day = df.index.min().normalize()
        df = df[df.index >= first_day + pd.Timedelta(days=1)]

    # persist cleaned data
    df.to_csv(out_csv)
    print("Cleaned data saved to:", out_csv)
    return df

# EDA plots
def plot_time_series(df: pd.DataFrame):
    ax = df["pollution"].plot(title="PM2.5 over time")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ts_pm25.png", dpi=160)
    plt.show()

    daily = df["pollution"].resample("D").mean()
    ax = daily.plot(title="PM2.5 (daily mean)")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ts_pm25_daily.png", dpi=160)
    plt.show()

def plot_missingness(df: pd.DataFrame):
    miss = df.isna().astype(int)
    plt.figure(figsize=(10, 3))
    plt.imshow(miss.T, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.xlabel("Time index")
    plt.title("Missingness map (1=NA)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missingness_map.png", dpi=160)
    plt.show()

    na_rate = df.isna().mean().sort_values(ascending=False)
    print("NA rates per column:\n", na_rate)

def plot_distributions(df: pd.DataFrame):
    # Histogram
    ax = df["pollution"].plot(kind="hist", bins=50, title="PM2.5 distribution")
    ax.set_xlabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dist_pm25_hist.png", dpi=160)
    plt.show()

    # Boxplot
    fig, ax = plt.subplots(figsize=(5, 5))
    df[["pollution"]].boxplot(ax=ax)
    ax.set_title("PM2.5 boxplot")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dist_pm25_box.png", dpi=160)
    plt.show()

    # Simple outlier summary using 1.5*IQR
    q1, q3 = df["pollution"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(0, q1 - 1.5 * iqr)
    n_out = ((df["pollution"] > upper) | (df["pollution"] < lower)).sum()
    print(f"IQR bounds: [{lower:.2f}, {upper:.2f}], # potential outliers: {n_out}")

def plot_seasonality(df: pd.DataFrame):
    # Diurnal pattern
    diurnal = df["pollution"].groupby(df.index.hour).mean()
    ax = diurnal.plot(marker="o", title="Diurnal mean PM2.5")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "season_diurnal.png", dpi=160)
    plt.show()

    # Weekly pattern
    weekly = df["pollution"].groupby(df.index.dayofweek).mean()
    ax = weekly.plot(marker="o", title="Weekly mean PM2.5 (0=Mon)")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "season_weekly.png", dpi=160)
    plt.show()

    # Monthly distributions
    month_vals = [df.loc[df.index.month == m, "pollution"].values for m in range(1, 13)]
    plt.figure(figsize=(10, 4))
    plt.boxplot(month_vals, labels=list(range(1, 13)), showfliers=False)
    plt.title("Monthly PM2.5 distribution")
    plt.xlabel("Month")
    plt.ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "season_monthly_box.png", dpi=160)
    plt.show()

    try:
        from statsmodels.tsa.seasonal import STL
        daily = df["pollution"].resample("D").mean().dropna()
        stl = STL(daily, period=7, robust=True).fit()
        stl.trend.plot(title="STL Trend"); plt.tight_layout()
        plt.savefig(FIG_DIR / "stl_trend.png", dpi=160); plt.show()
        stl.seasonal.plot(title="STL Seasonal (weekly)"); plt.tight_layout()
        plt.savefig(FIG_DIR / "stl_seasonal.png", dpi=160); plt.show()
        stl.resid.plot(title="STL Residual"); plt.tight_layout()
        plt.savefig(FIG_DIR / "stl_resid.png", dpi=160); plt.show()
    except Exception as e:
        print("[Info] STL decomposition skipped (install statsmodels to enable).", e)

def plot_correlations(df: pd.DataFrame):
    # numeric-only correlation
    num_cols = ["pollution","dew","temp","pressure","w_speed","snow","rain"]
    present = [c for c in num_cols if c in df.columns]
    corr = df[present].corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(present)), present, rotation=45, ha="right")
    plt.yticks(range(len(present)), present)
    plt.title("Correlation matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "corr_matrix.png", dpi=160)
    plt.show()

def plot_rolling_stats(df: pd.DataFrame):
    roll7 = df["pollution"].rolling(7*24, min_periods=24).mean()
    roll30 = df["pollution"].rolling(30*24, min_periods=24).mean()

    ax = df["pollution"].plot(alpha=0.4, label="hourly")
    roll7.plot(ax=ax, label="7-day mean")
    roll30.plot(ax=ax, label="30-day mean")
    ax.set_title("PM2.5 with rolling means")
    ax.set_ylabel("µg/m³")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_means.png", dpi=160)
    plt.show()

def plot_lag_behaviour(df: pd.DataFrame, lags=(1, 6, 12, 24, 48)):
    from pandas.plotting import lag_plot
    cols = len(lags)
    plt.figure(figsize=(3.2*cols, 3))
    for i, k in enumerate(lags, 1):
        plt.subplot(1, cols, i)
        # dropna to align
        sub = df[["pollution"]].copy()
        sub[f"lag{k}"] = sub["pollution"].shift(k)
        sub = sub.dropna()
        plt.scatter(sub[f"lag{k}"], sub["pollution"], s=5, alpha=0.5)
        plt.title(f"Lag {k}h")
        plt.xlabel(f"PM2.5(t-{k})"); plt.ylabel("PM2.5(t)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lag_scatter.png", dpi=160)
    plt.show()

def plot_wind_effects(df: pd.DataFrame):
    """Visualize how wind relates to PM2.5 (speed & direction)."""
    # Wind direction -> angle
    wdir = df["w_dir"].map(WIND_DEG)
    valid = wdir.notna() & df["w_speed"].notna() & df["pollution"].notna()
    if valid.sum() == 0:
        print("[Info] Wind effects skipped (no valid wind direction/speed).")
        return

    theta = np.deg2rad(wdir[valid].values)
    r = df.loc[valid, "w_speed"].values
    c = df.loc[valid, "pollution"].values

    # Polar scatter (rough wind-rose colored by PM2.5)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    sc = ax.scatter(theta, r, c=c, s=8, alpha=0.6)
    ax.set_title("Wind rose (colored by PM2.5)")
    plt.colorbar(sc, label="PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wind_rose_pm25.png", dpi=160)
    plt.show()

    # Speed vs PM2.5
    plt.figure(figsize=(6, 4))
    plt.scatter(df.loc[valid, "w_speed"], df.loc[valid, "pollution"], s=8, alpha=0.5)
    plt.title("Wind speed vs PM2.5")
    plt.xlabel("Wind speed"); plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wind_speed_vs_pm25.png", dpi=160)
    plt.show()

def quality_checks(df: pd.DataFrame):
    # Long flatlines (sensor stuck)
    flat = (df["pollution"].diff().abs() < 1e-12).rolling(24, min_periods=24).sum()
    stuck = flat[flat >= 24]
    print(f"Potential 24h flatline segments: {int((stuck >= 24).sum())}")

    # Value bounds sanity
    if (df["pollution"] > 1000).any():
        print("[Warn] Very large PM2.5 values detected (>1000 µg/m³). Check sensors/units.")

def single_column_view(df: pd.DataFrame):
    only = df[["pollution"]].copy()
    print(only.head())
    # Boxplot
    fig, ax = plt.subplots(figsize=(5, 5))
    only.boxplot(ax=ax)
    ax.set_title("PM2.5 boxplot (single column)")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "box_pm25_single.png", dpi=160)
    plt.show()


def main():
    df = load_and_clean(RAW_CSV, CLEAN_CSV)
    print(df.head())
    print(df.describe())


    plot_time_series(df)
    plot_missingness(df)
    plot_distributions(df)
    plot_seasonality(df)
    plot_correlations(df)
    plot_rolling_stats(df)
    plot_lag_behaviour(df)
    plot_wind_effects(df)
    quality_checks(df)
    single_column_view(df)

if __name__ == "__main__":
    main()
