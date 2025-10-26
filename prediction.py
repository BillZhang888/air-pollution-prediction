from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({
    "figure.figsize": (10, 4),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False
})
warnings.filterwarnings("ignore", category=UserWarning)

DATA_CLEAN = Path("pollution_clean.csv")
DATA_RAW   = Path("AirPollution.csv")
OUT_DIR    = Path("models_out")
FIG_DIR    = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

WIND_DEG = {
    "N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,"SE":135,"SSE":157.5,
    "S":180,"SSW":202.5,"SW":225,"WSW":247.5,"W":270,"WNW":292.5,"NW":315,"NNW":337.5
}

def _parse_dt(x: str) -> datetime:
    # for raw file 'year month day hour'
    return datetime.strptime(x, "%Y %m %d %H")

def load_data() -> pd.DataFrame:
    if DATA_CLEAN.exists():
        df = pd.read_csv(DATA_CLEAN, parse_dates=True, index_col=0)
        df = df.sort_index()
        return df

    if not DATA_RAW.exists():
        raise FileNotFoundError("Neither pollution_clean.csv nor AirPollution.csv was found.")

    orig_cols = ["No","year","month","day","hour","pm2.5","DEWP","TEMP","PRES","cbwd","Iws","Is","Ir"]
    new_cols  = ["pollution","dew","temp","pressure","w_dir","w_speed","snow","rain"]

    df = pd.read_csv(
        DATA_RAW,
        parse_dates=[["year","month","day","hour"]],
        date_parser=_parse_dt,
        index_col=0
    )
    if "No" in df.columns:
        df = df.drop(columns=["No"])
    df.columns = new_cols
    df = df.sort_index()

    for c in ["pollution","dew","temp","pressure","w_speed","snow","rain"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["pollution"] = df["pollution"].clip(lower=0).ffill().bfill().fillna(0)

    df["w_dir"] = df["w_dir"].astype(str).str.upper().str.strip()

    # drop the first day (consistent with earlier EDA)
    if len(df.index):
        first_day = df.index.min().normalize()
        df = df[df.index >= first_day + pd.Timedelta(days=1)]

    # save cleaned for reuse
    df.to_csv(DATA_CLEAN)
    return df

def encode_wind(df: pd.DataFrame) -> pd.DataFrame:
    wdeg = df["w_dir"].astype(str).str.upper().str.strip().map(WIND_DEG)
    rad = np.deg2rad(wdeg)
    df["w_sin"] = np.sin(rad)
    df["w_cos"] = np.cos(rad)
    # fill NaNs (unknown directions) with 0 (neutral)
    df[["w_sin","w_cos"]] = df[["w_sin","w_cos"]].fillna(0.0)
    return df

def add_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    hr = df.index.hour
    dow = df.index.dayofweek
    mon = df.index.month
    df["hour_sin"] = np.sin(2*np.pi*hr/24)
    df["hour_cos"] = np.cos(2*np.pi*hr/24)
    df["dow_sin"]  = np.sin(2*np.pi*dow/7)
    df["dow_cos"]  = np.cos(2*np.pi*dow/7)
    df["mon_sin"]  = np.sin(2*np.pi*mon/12)
    df["mon_cos"]  = np.cos(2*np.pi*mon/12)
    return df

def add_optional_lags(df: pd.DataFrame, lags=(1,)) -> pd.DataFrame:
    """Use previous hours if desired (kept minimal by default).
    NOTE: If you want strictly 'current-hour only' predictors, set lags=().
    """
    for k in lags:
        df[f"pollution_lag{k}"] = df["pollution"].shift(k)
    return df


def make_dataset(df: pd.DataFrame, use_lags=True):
    df = df.copy()
    df = encode_wind(df)
    df = add_time_cyclical(df)
    if use_lags:
        df = add_optional_lags(df, lags=(1,))  # add lag-1 as a strong short-term signal

    # target: next hour PM2.5
    df["y_next"] = df["pollution"].shift(-1)

    # drop last row (no label after shift) and any rows missing required fields
    df = df.dropna(subset=["y_next"])

    # feature list (numeric only; drop raw w_dir to avoid string→scaler error)
    base_feats = ["pollution","dew","temp","pressure","w_speed","snow","rain",
                  "w_sin","w_cos",
                  "hour_sin","hour_cos","dow_sin","dow_cos","mon_sin","mon_cos"]
    lag_feats = [c for c in df.columns if c.startswith("pollution_lag")]
    features = [c for c in base_feats + lag_feats if c in df.columns]

    X = df[features].copy()
    y = df["y_next"].copy()
    return X, y

def time_split(X, y, train=0.7, valid=0.15):
    n = len(X)
    i1 = int(train * n)
    i2 = int((train + valid) * n)
    X_tr, y_tr = X.iloc[:i1], y.iloc[:i1]
    X_va, y_va = X.iloc[i1:i2], y.iloc[i1:i2]
    X_te, y_te = X.iloc[i2:], y.iloc[i2:]
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def build_preprocessor(num_cols):
  
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", MinMaxScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )
    return pre

def build_selector(k="auto"):
    # placeholder; actual k is decided once we know n_features
    return SelectKBest(score_func=mutual_info_regression, k=k)


def model_linear(preprocessor, selector):
    return Pipeline([
        ("pre", preprocessor),
        ("sel", selector),
        ("est", LinearRegression())
    ])

def model_rf(preprocessor, selector):
    return Pipeline([
        ("pre", preprocessor),
        ("sel", selector),
        ("est", RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1
        ))
    ])


def evaluate_and_plot(name, pipe, X_tr, y_tr, X_va, y_va, X_te, y_te):
    # fit on train+valid (final fit)
    X_fit = pd.concat([X_tr, X_va])
    y_fit = pd.concat([y_tr, y_va])
    pipe.fit(X_fit, y_fit)

    pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred, squared=False)
    mape = (np.abs((y_te - pred) / np.clip(y_te, 1e-6, None))).mean() * 100
    r2 = r2_score(y_te, pred)

    print(f"\n== {name} (Test) ==")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²  : {r2:.3f}")

    # save predictions
    out_df = pd.DataFrame({"true": y_te.values, "pred": pred}, index=y_te.index)
    out_df.to_csv(OUT_DIR / f"{name.lower()}_pred.csv")

    # line plot (last 500 points for readability)
    k = min(500, len(out_df))
    seg = out_df.iloc[-k:]
    ax = seg["true"].reset_index(drop=True).plot(label="True")
    seg["pred"].reset_index(drop=True).plot(ax=ax, label="Pred")
    ax.set_title(f"{name} — True vs Pred (last {k} test points)")
    ax.set_xlabel("Test sample (time order)"); ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name.lower()}_ts.png", dpi=160); plt.show()

    # scatter with 45° line
    plt.figure(figsize=(5.5,5))
    plt.scatter(out_df["true"], out_df["pred"], s=10, alpha=0.5)
    mx = max(out_df["true"].max(), out_df["pred"].max()) * 1.05
    plt.plot([0, mx], [0, mx], "k--", lw=1)
    plt.title(f"{name} — True vs Pred (Test)")
    plt.xlabel("True PM2.5"); plt.ylabel("Predicted PM2.5")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name.lower()}_scatter.png", dpi=160); plt.show()

    # residuals
    resid = out_df["true"] - out_df["pred"]
    plt.figure(figsize=(10,3))
    plt.plot(resid.values, alpha=0.8)
    plt.title(f"{name} — Residuals (Test)")
    plt.xlabel("Test sample (time order)"); plt.ylabel("Residual")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name.lower()}_residuals.png", dpi=160); plt.show()

    plt.figure(figsize=(6,4))
    plt.hist(resid.values, bins=50)
    plt.title(f"{name} — Residual distribution")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name.lower()}_residual_hist.png", dpi=160); plt.show()

    # hour-of-day MAE
    err = np.abs(resid.values)
    tmp = pd.DataFrame({"err": err}, index=y_te.index)
    hour_mae = tmp.groupby(tmp.index.hour)["err"].mean()
    ax = hour_mae.plot(kind="bar")
    ax.set_title(f"{name} — MAE by Hour of Day (Test)")
    ax.set_xlabel("Hour"); ax.set_ylabel("MAE (µg/m³)")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name.lower()}_mae_by_hour.png", dpi=160); plt.show()

    # Feature importances (RF only)
    if name.lower() == "randomforest":
        # extract names after preprocessing + selection
        # We can approximate by running a small fit and tracing selector mask
        # Simpler: compute importances using permutation on test set (optional)
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=0, n_jobs=-1)
            # Map to input feature names:
            feat_names = X_te.columns.tolist()
            imp = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)
            imp.head(min(20, len(imp))).to_csv(OUT_DIR / "rf_permutation_importance.csv")
            top = imp.head(min(15, len(imp)))
            plt.figure(figsize=(7, max(3, 0.35*len(top)+1)))
            plt.barh(range(len(top))[::-1], top.values[::-1])
            plt.yticks(range(len(top))[::-1], top.index[::-1])
            plt.title("RandomForest — Permutation importance (Test)")
            plt.xlabel("Importance (mean decrease in score)")
            plt.tight_layout(); plt.savefig(FIG_DIR / "rf_perm_importance.png", dpi=160); plt.show()
        except Exception as e:
            print("[Info] Skipped permutation importance:", e)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def main():
    
    df = load_data()

    
    X, y = make_dataset(df, use_lags=True)


    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(X, y, train=0.7, valid=0.15)
    print(f"Samples -> Train: {len(X_tr)}, Valid: {len(X_va)}, Test: {len(X_te)}")

    num_cols = X.columns.tolist()
    pre = build_preprocessor(num_cols)

    k_auto = min(20, len(num_cols))
    sel = build_selector(k=k_auto)

    lin_pipe = model_linear(pre, sel)
    rf_pipe  = model_rf(pre, sel)

    scores_lin = evaluate_and_plot("Linear", lin_pipe, X_tr, y_tr, X_va, y_va, X_te, y_te)
    scores_rf  = evaluate_and_plot("RandomForest", rf_pipe, X_tr, y_tr, X_va, y_va, X_te, y_te)

    y_pred_persist = X_te["pollution"].values
    mae = mean_absolute_error(y_te, y_pred_persist)
    rmse = mean_squared_error(y_te, y_pred_persist, squared=False)
    mape = (np.abs((y_te - y_pred_persist) / np.clip(y_te, 1e-6, None))).mean() * 100
    r2 = r2_score(y_te, y_pred_persist)
    pd.DataFrame({
        "model": ["Persistence","Linear","RandomForest"],
        "MAE":   [mae, scores_lin["MAE"], scores_rf["MAE"]],
        "RMSE":  [rmse, scores_lin["RMSE"], scores_rf["RMSE"]],
        "MAPE%": [mape, scores_lin["MAPE"], scores_rf["MAPE"]],
        "R2":    [r2,  scores_lin["R2"],   scores_rf["R2"]],
    }).to_csv(OUT_DIR / "summary.csv", index=False)

    print("\nArtifacts saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
