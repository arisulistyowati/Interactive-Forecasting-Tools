"""
Streamlit App v2 — Generic Forecasting Pipeline (Univariate + Multivariate)

CSV minimum: 1 kolom tanggal + 1 kolom nilai (numerik).
Optional: kolom `level_name` (nation/region) untuk dukungan multi-region forecast.

Alur sequential:
    Step 1  Load & Data Preparation        (missing, dup, anomaly handling)
    Step 2  Exploratory Data Analysis      (variance, seasonality, trend)
    Step 3  Univariate Forecast            (forecast tiap kolom — target & tiap exog)
    Step 4  Multivariate Forecast          (target + exogs sbg regressor; future exog = hasil Step 3)

Mode:
    - Single (nation / no level_name): 1 series, 1 forecast.
    - Multi-region: dipilih dari kolom `level_name == 'region'`, forecast LOOP per region.

Run:
    streamlit run app.py
"""

import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from prophet import Prophet
from pymannkendall import original_test
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.seasonal import STL, seasonal_decompose

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Forecasting Pipeline v2 (Generic)",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "raw_df": None,
        "source_df": None,
        "prepared_df": None,
        # storage per (region, col) → nested {region: {col: data}}
        "uni_horizon_results": {},
        "uni_horizon_params":  {},
        "multi_horizon_results": {},   # {region: forecast_df}
        "multi_horizon_params":  {},   # {region: {start, end}}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================
# COLUMN AUTO-DETECT + NORMALIZE
# ============================================================
def autodetect_date_col(df):
    candidates = ["ds", "date", "tanggal", "time", "datetime", "timestamp"]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c:
                return col
    try:
        pd.to_datetime(df.iloc[:, 0])
        return df.columns[0]
    except Exception:
        return df.columns[0]


def autodetect_target_col(df, exclude):
    if "y" in df.columns:
        return "y"
    numeric_cols = [c for c in df.columns if c != exclude
                    and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])


def autodetect_location_col(df, exclude):
    candidates = ["location", "region", "ruas", "wilayah", "area", "branch", "site"]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c and col not in exclude:
                return col
    for col in df.columns:
        if col not in exclude and not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def normalize_df(df, date_col, value_cols, location_col=None):
    """Rename date_col → 'ds'. Jika location_col diberikan, pakai sbg 'location'; else 'all'."""
    keep_cols = [date_col] + value_cols + ([location_col] if location_col else [])
    out = df[keep_cols].copy()
    rename_map = {date_col: "ds"}
    if location_col:
        rename_map[location_col] = "location"
    out = out.rename(columns=rename_map)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if not location_col:
        out["location"] = "all"
    out["location"] = out["location"].astype(str)
    return out


# ============================================================
# DATA PREP
# ============================================================
def missing_values(data, value_cols):
    # Cleansing: nilai <= 0 dianggap invalid → jadikan NaN agar di-interpolasi
    for c in value_cols:
        data.loc[data[c] <= 0.0, c] = np.nan

    if "ds" in data.columns:
        data = data.set_index("ds")
    data.index = pd.to_datetime(data.index)
    out = []
    for region in data["location"].unique():
        sub = data[data["location"] == region].copy()
        for c in value_cols:
            sub[c] = sub[c].interpolate(method="time")
        out.append(sub)
    return pd.concat(out).sort_index()


def detect_anomalies_plot(series, period=52, z_threshold=3.0, model="additive"):
    decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    residual = decomposition.resid.dropna()
    z_scores = np.abs(stats.zscore(residual))
    mask = z_scores > z_threshold
    anomalies = series.loc[residual.index][mask]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.06,
        subplot_titles=["Data Asli + Anomali", "Trend", "Seasonal", "Residual"],
    )
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             name="Data Asli", line=dict(color="#38BDF8", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode="markers",
                             name=f"Anomali (n={mask.sum()})",
                             marker=dict(color="red", size=10, line=dict(color="white", width=1.2))),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, mode="lines",
                             name="Trend", line=dict(color="#A78BFA", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, mode="lines",
                             name="Seasonal", line=dict(color="#34D399", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=residual.index, y=residual.values, mode="lines",
                             name="Residual", line=dict(color="#94A3B8", width=1.2)), row=4, col=1)
    fig.update_layout(height=750, hovermode="x unified",
                      title=f"Anomaly Detection — period={period}, model={model}, z={z_threshold}")
    return fig, int(mask.sum())


def anomali_handling(data, col, periods=(12, 52)):
    if "ds" in data.columns:
        data = data.set_index("ds")
    data.index = pd.to_datetime(data.index)
    result_df, anomaly_dates_all = [], []
    for region in data["location"].unique():
        sub = data[data["location"] == region].copy()
        region_anom = []
        for period in periods:
            try:
                if len(sub[col].dropna()) < period * 2:
                    continue
                decomp = seasonal_decompose(sub[col], model="multiplicative", period=period)
                residual = decomp.resid.dropna()
                z_resid = (residual - residual.mean()) / residual.std()
                anomalies = residual[np.abs(z_resid) > 3]
                region_anom.extend(anomalies.index.tolist())
            except Exception:
                pass
        region_anom = sorted(set(region_anom))
        sub.loc[sub.index.isin(region_anom), col] = np.nan
        sub[col] = sub[col].interpolate(method="time")
        result_df.append(sub)
        anomaly_dates_all.extend(region_anom)
    return pd.concat(result_df).sort_index(), sorted(set(anomaly_dates_all))


# ============================================================
# EDA
# ============================================================
def cv_test(df, col, n_segment):
    y = df[col].dropna().values
    segments = np.array_split(y, n_segment)
    rows, cvs = [], []
    for i, seg in enumerate(segments):
        m, s = float(np.mean(seg)), float(np.std(seg))
        cv = s / m if m else np.nan
        cvs.append(cv)
        rows.append({"Segmen": f"Seg {i+1}", "Mean": round(m, 2),
                     "Std": round(s, 2), "CV": round(cv, 4)})
    cv_std = float(np.std(cvs))
    verdict = "MULTIPLICATIVE (CV stabil)" if cv_std < 0.05 else "ADDITIVE (CV tidak stabil)"
    return pd.DataFrame(rows), cv_std, verdict


def detect_strongest_seasonality(df, col):
    y = df.set_index("ds")[col].asfreq("D").interpolate()
    periods = {"weekly": 7, "monthly": 30, "yearly": 365}
    scores = {}
    for name, period in periods.items():
        if len(y) < period * 2:
            scores[name] = 0.0
            continue
        try:
            stl = STL(y, period=period, robust=True).fit()
            St, Rt = stl.seasonal, stl.resid
            var_R, var_SR = np.var(Rt, ddof=1), np.var(St + Rt, ddof=1)
            scores[name] = round(max(0.0, 1 - var_R / var_SR) if var_SR > 0 else 0.0, 4)
        except Exception:
            scores[name] = 0.0
    strongest = max(scores, key=scores.get)
    return scores, strongest


def check_trend(data, col, period, model):
    rows = []
    for loc in data["location"].unique():
        s = data[data["location"] == loc][col]
        try:
            res = seasonal_decompose(s, model=model, period=period)
            trend_series = res.trend.dropna()
            if len(trend_series):
                mk = original_test(trend_series.values)
                rows.append({"location": loc, "trend": mk.trend, "signifikan": mk.h,
                             "p-value": round(mk.p, 3), "z": round(mk.z, 3),
                             "Tau": round(mk.Tau, 3), "s": int(mk.s),
                             "slope": round(mk.slope, 3)})
            else:
                rows.append({"location": loc, "trend": None})
        except Exception as e:
            rows.append({"location": loc, "trend": f"error: {e}"})
    return pd.DataFrame(rows)


# ============================================================
# HOLIDAYS
# ============================================================
def parse_holidays(text):
    if not text or not text.strip():
        return None
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            ds = pd.to_datetime(parts[0])
        except Exception:
            continue
        name = parts[1] if len(parts) > 1 and parts[1] else "holiday"
        lo = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        hi = int(parts[3]) if len(parts) > 3 and parts[3] else 0
        rows.append({"holiday": name, "ds": ds, "lower_window": lo, "upper_window": hi})
    return pd.DataFrame(rows) if rows else None

def parse_anomaly_days(text):
    if not text or not text.strip():
        return None

    rows = []

    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < 2:
            continue

        try:
            start_date = pd.to_datetime(parts[0])
            end_date = pd.to_datetime(parts[1])
        except Exception:
            continue

        for ds in pd.date_range(start_date, end_date, freq="D"):
            rows.append({
                "holiday": "anomaly_days",
                "ds": ds,
                "lower_window": 0,
                "upper_window": 0
            })

    return pd.DataFrame(rows) if rows else None


# ============================================================
# PROPHET RUNNERS — UNIVARIATE
# ============================================================
@st.cache_data(show_spinner=False)
def run_eval_uni(data, target_col, holidays_df, season_mode, yearly, weekly, daily,
                 cp_range, n_cp, cps, spc, hps,
                 training_date, start_forecast, end_forecast):
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    df = data[["ds", target_col]].rename(columns={target_col: "y"}).dropna(subset=["y"])

    m = Prophet(
        yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily,
        seasonality_mode=season_mode, holidays=holidays_df,
        changepoint_range=cp_range, n_changepoints=n_cp,
        changepoint_prior_scale=cps, seasonality_prior_scale=spc, holidays_prior_scale=hps,
    )
    train = df[df["ds"] <= pd.to_datetime(training_date)]
    m.fit(train)
    future = pd.DataFrame({"ds": pd.date_range(start_forecast, end_forecast, freq="D")})
    forecast = m.predict(future)

    merged = pd.merge(df, forecast, on="ds", how="outer").sort_values("ds").reset_index(drop=True)
    merged["y_final"] = merged["yhat"].combine_first(merged["y"])

    actual = df[df["ds"].between(start_forecast, end_forecast)]
    predicted = forecast[forecast["ds"].between(start_forecast, end_forecast)][["ds", "yhat"]]
    eval_df = pd.merge(actual, predicted, on="ds", how="inner").dropna()
    mape = mean_absolute_percentage_error(eval_df["y"], eval_df["yhat"]) * 100 if len(eval_df) else None
    return merged, mape


@st.cache_data(show_spinner=False)
def run_horizon_uni(data, target_col, holidays_df, season_mode, yearly, weekly, daily,
                    cp_range, n_cp, cps, spc, hps,
                    start_forecast, end_forecast):
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    df = data[["ds", target_col]].rename(columns={target_col: "y"}).dropna(subset=["y"])

    m = Prophet(
        yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily,
        seasonality_mode=season_mode, holidays=holidays_df,
        changepoint_range=cp_range, n_changepoints=n_cp,
        changepoint_prior_scale=cps, seasonality_prior_scale=spc, holidays_prior_scale=hps,
    )
    m.fit(df)
    future = pd.DataFrame({"ds": pd.date_range(start_forecast, end_forecast, freq="D")})
    forecast = m.predict(future)
    merged = pd.merge(df, forecast, on="ds", how="outer").sort_values("ds").reset_index(drop=True)
    merged["y_final"] = merged["y"].combine_first(merged["yhat"])
    return merged


# ============================================================
# PROPHET RUNNERS — MULTIVARIATE
# ============================================================
@st.cache_data(show_spinner=False)
def run_eval_multi(data, target_col, exog_cols, holidays_df, season_mode, yearly, weekly, daily,
                   cp_range, n_cp, cps, spc, hps,
                   training_date, start_forecast, end_forecast):
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    df = data[["ds", target_col] + exog_cols].rename(columns={target_col: "y"})

    m = Prophet(
        yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily,
        seasonality_mode=season_mode, holidays=holidays_df,
        changepoint_range=cp_range, n_changepoints=n_cp,
        changepoint_prior_scale=cps, seasonality_prior_scale=spc, holidays_prior_scale=hps,
    )

    # ➜ Yearly seasonality lebih fleksibel
    m.add_seasonality(name='yearly_custom', period=365.25, fourier_order=20, mode='multiplicative')

    # ➜ Workweek seasonality
    m.add_seasonality(name="weekly", period=7, fourier_order=8, prior_scale=12, mode="multiplicative")

    for ex in exog_cols:
        m.add_regressor(ex)

    train = df[df["ds"] <= pd.to_datetime(training_date)].dropna(subset=["y"] + exog_cols)
    m.fit(train)

    future = pd.DataFrame({"ds": pd.date_range(start_forecast, end_forecast, freq="D")})
    future = pd.merge(future, df[["ds"] + exog_cols], on="ds", how="left")
    forecast = m.predict(future.dropna(subset=exog_cols))

    merged = pd.merge(df[["ds", "y"]], forecast, on="ds", how="outer"
                      ).sort_values("ds").reset_index(drop=True)
    merged["y_final"] = merged["yhat"].combine_first(merged["y"])

    actual = df[df["ds"].between(start_forecast, end_forecast)][["ds", "y"]]
    predicted = forecast[forecast["ds"].between(start_forecast, end_forecast)][["ds", "yhat"]]
    eval_df = pd.merge(actual, predicted, on="ds", how="inner").dropna()
    mape = mean_absolute_percentage_error(eval_df["y"], eval_df["yhat"]) * 100 if len(eval_df) else None
    return merged, mape


@st.cache_data(show_spinner=False)
def run_horizon_multi(data, target_col, exog_cols, future_exog_df,
                      holidays_df, season_mode, yearly, weekly, daily,
                      cp_range, n_cp, cps, spc, hps,
                      start_forecast, end_forecast):
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    df = data[["ds", target_col] + exog_cols].rename(columns={target_col: "y"})

    m = Prophet(
        yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily,
        seasonality_mode=season_mode, holidays=holidays_df,
        changepoint_range=cp_range, n_changepoints=n_cp,
        changepoint_prior_scale=cps, seasonality_prior_scale=spc, holidays_prior_scale=hps,
    )

    # ➜ Yearly seasonality lebih fleksibel
    m.add_seasonality(name='yearly_custom', period=365.25, fourier_order=20, mode='multiplicative')

    # ➜ Workweek seasonality
    m.add_seasonality(name="weekly", period=7, fourier_order=8, prior_scale=12, mode="multiplicative")

    for ex in exog_cols:
        m.add_regressor(ex)
    m.fit(df.dropna(subset=["y"] + exog_cols))

    future = pd.DataFrame({"ds": pd.date_range(start_forecast, end_forecast, freq="D")})
    future = pd.merge(future, future_exog_df[["ds"] + exog_cols], on="ds", how="left")
    forecast = m.predict(future.dropna(subset=exog_cols))

    merged = pd.merge(df[["ds", "y"]], forecast, on="ds", how="outer"
                      ).sort_values("ds").reset_index(drop=True)
    merged["y_final"] = merged["y"].combine_first(merged["yhat"])
    return merged


# ============================================================
# PLOT HELPERS
# ============================================================
def plot_actual(df, col, title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df[col], mode="lines+markers",
                             line=dict(color="#1f77b4"), name=col))
    fig.update_layout(title=title or f"Actual {col}", xaxis_title="Tanggal",
                      yaxis_title=col, template="plotly_white", height=400)
    return fig


def plot_forecast(df_forecast, df_actual=None, title="Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"], y=df_forecast["yhat_upper"],
        mode="lines", line=dict(width=0, color="rgba(70,130,180,0.4)"),
        name="yhat_upper",
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"], y=df_forecast["yhat_lower"],
        mode="lines", line=dict(width=0, color="rgba(70,130,180,0.4)"),
        fill="tonexty", fillcolor="rgba(70,130,180,0.20)",
        name="yhat_lower",
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"], y=df_forecast["yhat"],
        mode="lines", line=dict(color="steelblue", width=2.8), name="yhat",
    ))
    if df_actual is not None and len(df_actual):
        fig.add_trace(go.Scatter(
            x=df_actual["ds"], y=df_actual["y"], mode="lines+markers",
            line=dict(color="tomato", width=1.6), marker=dict(size=4),
            name="Actual",
        ))
        cutoff = pd.to_datetime(df_actual["ds"].max())
        fig.add_shape(type="line",
                      x0=cutoff, x1=cutoff, xref="x",
                      y0=0, y1=1, yref="paper",
                      line=dict(color="grey", width=1.5, dash="dash"))
        fig.add_annotation(x=cutoff, y=1, xref="x", yref="paper",
                           text="Forecast Start", showarrow=False,
                           yanchor="bottom", font=dict(color="grey", size=11))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value",
                      template="plotly_white", height=500, hovermode="x unified")
    return fig


def prophet_param_block(prefix, default_cpr, default_cps):
    c1, c2, c3, c4, c5 = st.columns(5)
    cpr = c1.number_input("changepoint_range",       0.0,   1.0,  default_cpr, 0.05, key=f"{prefix}_cpr")
    ncp = c2.number_input("n_changepoints",          1,     100,  25,           1,   key=f"{prefix}_ncp")
    cps = c3.number_input("changepoint_prior_scale", 0.001, 1.0,  default_cps,  0.005, format="%.3f", key=f"{prefix}_cps")
    spc = c4.number_input("seasonality_prior_scale", 0.01,  50.0, 10.0,         0.5, key=f"{prefix}_spc")
    hps = c5.number_input("holidays_prior_scale",    0.01,  50.0, 10.0,         0.5, key=f"{prefix}_hps")
    c6, c7, c8, c9 = st.columns(4)
    smode  = c6.selectbox("seasonality_mode", ["multiplicative", "additive"], key=f"{prefix}_smd")
    yearly = c7.checkbox("yearly_seasonality", True, key=f"{prefix}_y")
    weekly = c8.checkbox("weekly_seasonality", True, key=f"{prefix}_w")
    daily  = c9.checkbox("daily_seasonality",  False, key=f"{prefix}_d")
    return dict(cpr=cpr, ncp=ncp, cps=cps, spc=spc, hps=hps,
                smode=smode, yearly=yearly, weekly=weekly, daily=daily)


# ============================================================
# SIDEBAR — DATA & COLUMN SELECTION
# ============================================================
st.sidebar.title("📂 Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sep = st.sidebar.selectbox("Delimiter", [",", ";", "\\t", "|"], index=0)

st.title("📈 Forecasting Pipeline v2 — Generic (Univariate + Multivariate)")
st.caption(
    "Upload **satu CSV**. Pilih **level_name** (nation/region) untuk single vs multi-region forecast. "
    "Alur: Prep → EDA → Univariate (per kolom & region) → Multivariate (target + exogs)."
)

if uploaded is None:
    st.warning("⬅️ Upload satu file CSV di sidebar untuk memulai.")
    st.markdown(
        "**Format CSV minimum:** 1 kolom tanggal + ≥1 kolom nilai numerik.  \n\n"
        "**Untuk multi-region forecast**, tambahkan kolom `level_name` (nilai: `nation` / `region`) "
        "dan kolom region/location.  \n\n"
        "Contoh single (nation):\n```\ntanggal,level_name,nilai\n2023-01-01,nation,100\n2023-01-02,nation,105\n```\n"
        "Contoh multi-region:\n```\ntanggal,level_name,region,kendaraan,pendapatan\n"
        "2023-01-01,region,Jakarta,1200,500000\n2023-01-01,region,Surabaya,900,380000\n```"
    )
    st.stop()

try:
    delimiter = "\t" if sep == "\\t" else sep
    df_raw = pd.read_csv(uploaded, sep=delimiter)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded: {df_raw.shape[0]} baris × {df_raw.shape[1]} kolom")

# --- Date column
date_default = autodetect_date_col(df_raw)
date_col = st.sidebar.selectbox("Kolom tanggal", df_raw.columns,
                                index=list(df_raw.columns).index(date_default))

# --- level_name filter (opsional)
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Level Filter")

level_col_name = None
for col in df_raw.columns:
    if col.lower() == "level_name":
        level_col_name = col
        break

selected_level = None
location_col_csv = None
selected_regions = None
is_multi_region = False

if level_col_name:
    levels_avail = sorted(df_raw[level_col_name].dropna().astype(str).unique().tolist())
    selected_level = st.sidebar.selectbox(
        f"Pilih `{level_col_name}`",
        levels_avail,
        help="Kategori data: `nation` = single series, `region` = multi-region (forecast loop per region).",
    )
    df_raw = df_raw[df_raw[level_col_name].astype(str) == selected_level].copy()
    st.sidebar.caption(f"Filtered: {df_raw.shape[0]} baris setelah filter `{selected_level}`.")

    if selected_level.lower() == "region":
        is_multi_region = True
        # detect location/region column
        loc_default = autodetect_location_col(df_raw, exclude=[date_col, level_col_name])
        text_like_cols = [c for c in df_raw.columns
                          if c not in (date_col, level_col_name)
                          and not pd.api.types.is_numeric_dtype(df_raw[c])]
        if not text_like_cols:
            text_like_cols = [c for c in df_raw.columns if c not in (date_col, level_col_name)]
        idx_default = text_like_cols.index(loc_default) if loc_default in text_like_cols else 0
        location_col_csv = st.sidebar.selectbox(
            "Kolom region",
            text_like_cols,
            index=idx_default,
        )
        all_regions = sorted(df_raw[location_col_csv].dropna().astype(str).unique().tolist())
        selected_regions = st.sidebar.multiselect(
            f"Region utk forecast ({len(all_regions)} tersedia)",
            all_regions, default=all_regions,
        )
        if not selected_regions:
            st.sidebar.warning("Pilih minimal 1 region.")
            st.stop()
        df_raw = df_raw[df_raw[location_col_csv].astype(str).isin(selected_regions)]
else:
    st.sidebar.info("Kolom `level_name` tidak ditemukan — mode single series.")

# --- Numeric target & exog
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Target & Exog")

excluded = {date_col}
if level_col_name: excluded.add(level_col_name)
if location_col_csv: excluded.add(location_col_csv)

numeric_cols = [c for c in df_raw.columns if c not in excluded
                and pd.api.types.is_numeric_dtype(df_raw[c])]
if not numeric_cols:
    numeric_cols = [c for c in df_raw.columns if c not in excluded]

target_default = autodetect_target_col(df_raw, exclude=date_col)
if target_default not in numeric_cols and numeric_cols:
    target_default = numeric_cols[0]

target_col = st.sidebar.selectbox(
    "Kolom TARGET (yg di-forecast)",
    numeric_cols,
    index=numeric_cols.index(target_default) if target_default in numeric_cols else 0,
)

available_exogs = [c for c in numeric_cols if c != target_col]
exog_cols = st.sidebar.multiselect(
    "Kolom EXOG (opsional, untuk multivariate)",
    available_exogs, default=[],
)

all_value_cols = [target_col] + exog_cols

# --- Normalize
df_norm = normalize_df(df_raw, date_col, all_value_cols, location_col=location_col_csv)
if df_norm.empty:
    st.error("Setelah parsing tanggal, data kosong. Cek kolom tanggal yang dipilih.")
    st.stop()

# regions_list: list region aktif (yg dipakai untuk loop)
if is_multi_region:
    regions_list = sorted(df_norm["location"].unique().tolist())
else:
    regions_list = sorted(df_norm["location"].unique().tolist())  # biasanya ['all']

date_min = df_norm["ds"].min().date()
date_max = df_norm["ds"].max().date()
st.sidebar.caption(f"Rentang tanggal: **{date_min}** s/d **{date_max}**")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Training cutoff")
training_cutoff = st.sidebar.date_input(
    "Training cutoff", value=date_max, min_value=date_min, max_value=date_max,
    help="Data ≤ cutoff = training. Data > cutoff dipakai utk MAPE Actual."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎉 Holiday (opsional)")
holidays_text = st.sidebar.text_area(
    "Format per baris: `YYYY-MM-DD,nama,lower,upper`",
    value="",
    height=120,
    key="holidays_text",
    help="Contoh:\n2024-04-10,idul_fitri,-7,7\n2024-12-25,natal,0,0",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Anomaly Days (opsional)")

anomalies_text = st.sidebar.text_area(
    "Format per baris: `start_date,end_date`",
    value="",
    height=120,
    key="anomalies_text",
    help="Contoh:\n2025-01-17,2025-06-28\n2025-08-01,2025-08-10",
)

holiday_df = parse_holidays(holidays_text)
anomaly_df = parse_anomaly_days(anomalies_text)

frames = []

if holiday_df is not None:
    frames.append(holiday_df)

if anomaly_df is not None:
    frames.append(anomaly_df)

holidays_df = pd.concat(frames, ignore_index=True) if frames else None


# Banner
mode_label = "MULTIVARIATE" if exog_cols else "UNIVARIATE-ONLY"
region_label = f"{len(regions_list)} region (loop)" if is_multi_region else "single series"
st.info(
    f"**Mode:** {mode_label} · {region_label}  |  **Level:** `{selected_level or '—'}`  |  "
    f"**Target:** `{target_col}`  |  **Exogs:** {', '.join(f'`{e}`' for e in exog_cols) if exog_cols else '_(none)_'}"
)

raw_df = df_norm.copy()
source_df = df_norm[df_norm["ds"] <= pd.to_datetime(training_cutoff)].copy()
st.session_state.raw_df = raw_df
st.session_state.source_df = source_df


# ============================================================
# UTIL — region focus selector (di tiap tab)
# ============================================================
def region_focus_selector(key, default_region=None):
    """Selectbox untuk fokus 1 region (multi-region) atau noop (single)."""
    if not is_multi_region:
        return regions_list[0]  # 'all'
    default_idx = 0
    if default_region and default_region in regions_list:
        default_idx = regions_list.index(default_region)
    return st.selectbox("📍 Region focus", regions_list, index=default_idx, key=key)


def slice_region(df, region):
    return df[df["location"].astype(str) == str(region)].copy()


# ============================================================
# TABS
# ============================================================
tab_labels = [
    "1️⃣ Load & Preparation",
    "2️⃣ EDA",
    "3️⃣ Univariate Forecast",
    "4️⃣ Multivariate Forecast",
]
tab1, tab2, tab3, tab4 = st.tabs(tab_labels)


# ----------------------------------------------------------------
# TAB 1 — LOAD & PREP
# ----------------------------------------------------------------
with tab1:
    st.header("Step 1 — Load & Data Preparation")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("source_df (training)")
        st.dataframe(source_df.head(), use_container_width=True)
        st.caption(f"Shape: {source_df.shape}  |  Range: {source_df['ds'].min().date()} → {source_df['ds'].max().date()}")
    with c2:
        st.subheader("raw_df (full)")
        st.dataframe(raw_df.head(), use_container_width=True)
        st.caption(f"Shape: {raw_df.shape}  |  Range: {raw_df['ds'].min().date()} → {raw_df['ds'].max().date()}")

    focus_region_t1 = region_focus_selector("focus_region_t1")
    preview_col = st.selectbox("Preview kolom", all_value_cols, key="prev_col")
    preview_data = slice_region(raw_df, focus_region_t1) if is_multi_region else raw_df
    st.plotly_chart(
        plot_actual(preview_data, preview_col,
                    title=f"Raw data — {preview_col}" + (f" · region={focus_region_t1}" if is_multi_region else "")),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("1.2 Handling Missing Values (interpolasi `time`, semua kolom nilai)")

    miss_before = source_df[all_value_cols].isnull().sum()
    st.write("Missing **sebelum** handling:", miss_before.to_dict())

    mvh = missing_values(source_df.copy(), all_value_cols).reset_index()
    miss_after = mvh[all_value_cols].isnull().sum()
    st.write("Missing **setelah** handling:", miss_after.to_dict())

    st.markdown("---")
    st.subheader("1.3 Check Duplicated")

    dup_count = int(source_df.duplicated(subset=["ds", "location"]).sum())
    st.metric("Duplikat (ds × location)", dup_count)
    if dup_count > 0:
        st.warning("Ada duplikat — agregasi dulu sebelum forecasting.")

    st.markdown("---")

    # ----------------------------------------------------------------
    # 1.4 Handling Anomalies
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("1.4 Handling Anomalies")

    cAnA, cAnB, cAnC, cAnD = st.columns(4)

    anom_col = cAnA.selectbox(
        "Kolom utk visualisasi anomali",
        all_value_cols,
        key="anom_col"
    )

    anom_period = cAnB.number_input(
        "Period (seasonal)",
        min_value=2,
        max_value=730,
        value=52,
        step=1
    )

    anom_z = cAnC.number_input(
        "Z-threshold",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.1
    )

    anom_model = cAnD.selectbox(
        "Model decompose",
        ["additive", "multiplicative"]
    )

    n_iter = st.number_input(
        "Jumlah iterasi anomaly handling",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

    # Inisialisasi session_state
    if "prepared_df" not in st.session_state:
        st.session_state.prepared_df = None

    if "iteration_summary" not in st.session_state:
        st.session_state.iteration_summary = []

    # ==========================================================
    # RUN
    # ==========================================================
    if st.button("🔍 Run Anomaly Detection & Handling", type="primary"):

        # ------------------------------------------------------
        # Visualisasi Anomali
        # ------------------------------------------------------
        mvh_focus = slice_region(mvh, focus_region_t1)

        mvh_idx = mvh_focus.set_index("ds")
        mvh_idx.index = pd.to_datetime(mvh_idx.index)

        with st.spinner(
            f"Detecting anomalies pada `{anom_col}` "
            f"(region={focus_region_t1})..."
        ):
            try:
                fig_anom, n_anom = detect_anomalies_plot(
                    mvh_idx[anom_col],
                    period=int(anom_period),
                    z_threshold=float(anom_z),
                    model=anom_model,
                )

                st.plotly_chart(fig_anom, use_container_width=True)

                st.success(
                    f"`{anom_col}` @ region "
                    f"`{focus_region_t1}` : "
                    f"**{n_anom}** anomali terdeteksi."
                )

            except Exception as e:
                st.error(f"Gagal decompose: {e}")

        # ------------------------------------------------------
        # Handling Anomali Berulang
        # ------------------------------------------------------
        with st.spinner(
            f"Running anomaly handling ({n_iter} iterasi)..."
        ):

            prepared = mvh.copy()

            iteration_summary = []

            for i in range(n_iter):

                iter_result = {}

                for col in all_value_cols:

                    handled, dates = anomali_handling(
                        prepared.copy(),
                        col
                    )

                    handled = handled.reset_index()

                    prepared = prepared.merge(
                        handled[
                            ["ds", "location", col]
                        ].rename(
                            columns={col: f"_new_{col}"}
                        ),
                        on=["ds", "location"],
                        how="left"
                    )

                    prepared[col] = prepared[f"_new_{col}"]

                    prepared.drop(
                        columns=[f"_new_{col}"],
                        inplace=True
                    )

                    iter_result[col] = len(dates)

                iteration_summary.append(iter_result)

            prepared["ds"] = pd.to_datetime(prepared["ds"])

            # Simpan hasil akhir
            st.session_state.prepared_df = prepared

            # Simpan ringkasan iterasi
            st.session_state.iteration_summary = iteration_summary

            # Reset downstream result
            st.session_state.uni_horizon_results = {}
            st.session_state.uni_horizon_params = {}

            st.session_state.multi_horizon_results = {}
            st.session_state.multi_horizon_params = {}


    # ==========================================================
    # TAMPILKAN HASIL AKHIR
    # ==========================================================
    if st.session_state.prepared_df is not None:

        st.subheader(
            f"After Handling "
            f"({len(st.session_state.iteration_summary)} Iterasi)"
        )

        st.plotly_chart(
            plot_actual(
                slice_region(
                    st.session_state.prepared_df,
                    focus_region_t1
                ),
                anom_col,
                title=(
                    f"After Anomaly Handling "
                    f"({len(st.session_state.iteration_summary)} Iterasi)"
                    f" — {anom_col}"
                ),
            ),
            use_container_width=True,
        )

        st.success(
            f"✅ Selesai. Hasil akhir disimpan pada "
            f"`st.session_state.prepared_df`."
        )


# ----------------------------------------------------------------
# TAB 2 — EDA
# ----------------------------------------------------------------
with tab2:
    st.header("Step 2 — Exploratory Data Analysis")

    if st.session_state.prepared_df is None:
        st.warning("Jalankan Step 1 terlebih dahulu.")
        st.stop()

    prepared_full = st.session_state.prepared_df
    focus_region_t2 = region_focus_selector("focus_region_t2")
    prepared = slice_region(prepared_full, focus_region_t2) if is_multi_region else prepared_full

    eda_col = st.selectbox("Kolom untuk analisis", all_value_cols, key="eda_col")

    # 2.1 Variance
    st.subheader(f"2.1 Variance Stability — `{eda_col}` · region={focus_region_t2}")
    n_seg = st.slider("Jumlah segmen", 2, 10, 4)
    cv_df, cv_std, verdict = cv_test(prepared, eda_col, n_seg)
    st.dataframe(cv_df, use_container_width=True)
    cA, cB = st.columns(2)
    cA.metric("Std of CV antar segmen", f"{cv_std:.4f}")
    cB.metric("Verdict", verdict)

    st.markdown("---")
    # 2.2 Seasonality
    st.subheader(f"2.2 Strongest Seasonality — `{eda_col}` · region={focus_region_t2}")
    scores, strongest = detect_strongest_seasonality(prepared, eda_col)
    cols = st.columns(len(scores) + 1)
    for i, (name, val) in enumerate(scores.items()):
        cols[i].metric(name.capitalize(), f"{val:.4f}",
                       delta="STRONGEST" if name == strongest else None)
    cols[-1].metric("Weekly Mode", str(strongest == "weekly"))

    fig_seas = go.Figure(go.Bar(
        x=list(scores.keys()), y=list(scores.values()),
        marker_color=["#34D399" if k == strongest else "#94A3B8" for k in scores],
        text=[f"{v:.3f}" for v in scores.values()], textposition="outside",
    ))
    fig_seas.update_layout(title=f"Seasonality Strength — {eda_col} · {focus_region_t2}",
                           yaxis_title="FS score", template="plotly_white", height=350)
    st.plotly_chart(fig_seas, use_container_width=True)

    st.markdown("---")
    # 2.3 Trend
    st.subheader(f"2.3 Check Trend (Mann-Kendall) — `{eda_col}`")
    cT1, cT2 = st.columns(2)
    trend_period = cT1.number_input("Period (seasonal_decompose)", 2, 730, 365)
    trend_model  = cT2.selectbox("Model", ["multiplicative", "additive"], key="trend_md")
    # untuk multi-region: jalankan utk semua region biar bisa dibandingkan
    src_trend = prepared_full if is_multi_region else prepared
    trend_df = check_trend(src_trend, eda_col, int(trend_period), trend_model)
    st.dataframe(trend_df, use_container_width=True)


# ----------------------------------------------------------------
# TAB 3 — UNIVARIATE FORECAST
# ----------------------------------------------------------------
with tab3:
    st.header("Step 3 — Univariate Forecast")
    st.caption(
        "Forecast tiap kolom. Multi-region → otomatis **loop per region**. "
        "Untuk Step 4 multivariate, setiap exog harus punya **Horizon Forecast** di tiap region (dengan rentang yg sama)."
    )

    if st.session_state.prepared_df is None:
        st.warning("Jalankan Step 1 terlebih dahulu.")
        st.stop()

    prepared_full = st.session_state.prepared_df
    raw_local     = st.session_state.raw_df
    p_min, p_max  = prepared_full["ds"].min().date(), prepared_full["ds"].max().date()

    fc_col = st.selectbox(
        "Pilih kolom untuk forecast univariate",
        all_value_cols,
        index=0,
        help="Forecast independen per kolom (target & tiap exog).",
    )

    # Indikator: region × kolom yg sudah punya horizon forecast
    done_summary = []
    for r in regions_list:
        cols_done = list(st.session_state.uni_horizon_results.get(r, {}).keys())
        if cols_done:
            done_summary.append(f"`{r}`: {', '.join(cols_done)}")
    if done_summary:
        st.caption("✅ Horizon forecast sudah dijalankan utk → " + " · ".join(done_summary))

    focus_region_t3 = region_focus_selector("focus_region_t3")

    # === 3.A MAPE MODEL ===
    st.subheader(f"3.A MAPE Model — `{fc_col}`")
    with st.form(key=f"form_mape_uni_{fc_col}"):
        with st.expander("⚙️ Parameter MAPE Model", expanded=True):
            c1, c2, c3 = st.columns(3)
            e_train = c1.date_input("Training cutoff",
                                    value=p_max - pd.Timedelta(days=90),
                                    min_value=p_min, max_value=p_max, key=f"e_tr_{fc_col}")
            e_start = c2.date_input("Start forecast",
                                    value=p_max - pd.Timedelta(days=89),
                                    min_value=p_min, max_value=p_max, key=f"e_st_{fc_col}")
            e_end   = c3.date_input("End forecast",
                                    value=p_max,
                                    min_value=p_min, max_value=p_max, key=f"e_en_{fc_col}")
            params_e = prophet_param_block(f"e_{fc_col}", default_cpr=0.5, default_cps=0.01)
        submitted_mape_uni = st.form_submit_button("🚀 Run MAPE Model")

    if submitted_mape_uni:
        mape_rows = []
        plots_eval = {}
        progress = st.progress(0.0, text="Forecasting per region...")
        for i, region in enumerate(regions_list):
            sub_prep = slice_region(prepared_full, region)
            if sub_prep.empty:
                continue
            try:
                exec_df, mape_val = run_eval_uni(
                    sub_prep, fc_col, holidays_df,
                    params_e["smode"], params_e["yearly"], params_e["weekly"], params_e["daily"],
                    float(params_e["cpr"]), int(params_e["ncp"]), float(params_e["cps"]),
                    float(params_e["spc"]), float(params_e["hps"]),
                    str(e_train), str(e_start), str(e_end),
                )
                mape_rows.append({"region": region, "mape (%)": round(mape_val, 3) if mape_val else None})
                plots_eval[region] = exec_df
            except Exception as ex:
                mape_rows.append({"region": region, "mape (%)": f"error: {ex}"})
            progress.progress((i + 1) / len(regions_list), text=f"Region {region} selesai")
        progress.empty()

        st.subheader("Hasil MAPE Model per Region")
        st.dataframe(pd.DataFrame(mape_rows), use_container_width=True)

        # Plot region terfokus
        if focus_region_t3 in plots_eval:
            exec_df = plots_eval[focus_region_t3]
            df_fc = exec_df[exec_df["ds"] >= pd.to_datetime(e_start)]
            df_ac = exec_df[exec_df["ds"] <= pd.to_datetime(e_train)][["ds", "y"]].dropna()
            mv = next((r["mape (%)"] for r in mape_rows if r["region"] == focus_region_t3), None)
            title = f"Univariate MAPE Model — {fc_col} · {focus_region_t3}" + (f" | MAPE = {mv} %" if mv else "")
            st.plotly_chart(plot_forecast(df_fc, df_ac, title=title), use_container_width=True)

        if is_multi_region:
            with st.expander("Plot per region (semua)"):
                for region, exec_df in plots_eval.items():
                    df_fc = exec_df[exec_df["ds"] >= pd.to_datetime(e_start)]
                    df_ac = exec_df[exec_df["ds"] <= pd.to_datetime(e_train)][["ds", "y"]].dropna()
                    mv = next((r["mape (%)"] for r in mape_rows if r["region"] == region), None)
                    st.markdown(f"**Region: `{region}`** — MAPE = {mv}")
                    st.plotly_chart(
                        plot_forecast(df_fc, df_ac, title=f"{fc_col} · {region}"),
                        use_container_width=True,
                    )

    st.markdown("---")

    # === 3.B HORIZON ===
    st.subheader(f"3.B Horizon Forecast — `{fc_col}`")
    with st.form(key=f"form_horizon_uni_{fc_col}"):
        with st.expander("⚙️ Parameter Horizon Forecast", expanded=True):
            c1, c2 = st.columns(2)
            h_start = c1.date_input("Start horizon",
                                    value=p_max + pd.Timedelta(days=1), key=f"h_st_{fc_col}")
            h_end   = c2.date_input("End horizon",
                                    value=p_max + pd.Timedelta(days=180), key=f"h_en_{fc_col}")
            params_h = prophet_param_block(f"h_{fc_col}", default_cpr=0.8, default_cps=0.05)
        submitted_horizon_uni = st.form_submit_button("🚀 Run Horizon Forecast", type="primary")

    if submitted_horizon_uni:
        progress = st.progress(0.0, text="Forecasting horizon per region...")
        for i, region in enumerate(regions_list):
            sub_prep = slice_region(prepared_full, region)
            if sub_prep.empty:
                continue
            try:
                horizon_result = run_horizon_uni(
                    sub_prep, fc_col, holidays_df,
                    params_h["smode"], params_h["yearly"], params_h["weekly"], params_h["daily"],
                    float(params_h["cpr"]), int(params_h["ncp"]), float(params_h["cps"]),
                    float(params_h["spc"]), float(params_h["hps"]),
                    str(h_start), str(h_end),
                )
                st.session_state.uni_horizon_results.setdefault(region, {})[fc_col] = horizon_result
                st.session_state.uni_horizon_params.setdefault(region, {})[fc_col] = dict(
                    start=str(h_start), end=str(h_end),
                )
            except Exception as ex:
                st.error(f"Region `{region}` gagal: {ex}")
            progress.progress((i + 1) / len(regions_list), text=f"Region {region} selesai")
        progress.empty()
        st.success(f"✅ Horizon forecast `{fc_col}` tersimpan utk {len(regions_list)} region.")

    # Display: focused region
    if (focus_region_t3 in st.session_state.uni_horizon_results
            and fc_col in st.session_state.uni_horizon_results[focus_region_t3]):
        hr = st.session_state.uni_horizon_results[focus_region_t3][fc_col]
        params = st.session_state.uni_horizon_params[focus_region_t3][fc_col]
        df_fc = hr[hr["ds"] >= pd.to_datetime(params["start"])]
        df_ac = hr[hr["ds"] < pd.to_datetime(params["start"])][["ds", "y"]].dropna()
        st.plotly_chart(
            plot_forecast(df_fc, df_ac,
                          title=f"Horizon — {fc_col} · {focus_region_t3} ({params['start']} → {params['end']})"),
            use_container_width=True,
        )

        # === 3.C MAPE ACTUAL ===
        st.markdown("---")
        st.subheader(f"3.C MAPE Actual — `{fc_col}`")
        mape_actual_rows = []
        for region in regions_list:
            if region not in st.session_state.uni_horizon_results:
                continue
            if fc_col not in st.session_state.uni_horizon_results[region]:
                continue
            hr_r = st.session_state.uni_horizon_results[region][fc_col]
            pr   = st.session_state.uni_horizon_params[region][fc_col]
            forecast_h = hr_r.loc[
                hr_r["ds"].between(pr["start"], pr["end"]),
                ["ds", "yhat"]
            ].rename(columns={"yhat": "forecast"})
            raw_r = slice_region(raw_local, region)
            actual_h = raw_r.loc[
                raw_r["ds"].between(pr["start"], pr["end"]),
                ["ds", fc_col]
            ].rename(columns={fc_col: "actual"})
            eval_h = pd.merge(actual_h, forecast_h, on="ds", how="inner").dropna()
            if len(eval_h):
                mape_actual = mean_absolute_percentage_error(eval_h["actual"], eval_h["forecast"]) * 100
                mape_actual_rows.append({"region": region, "n_points": len(eval_h),
                                         "mape_actual (%)": round(mape_actual, 3)})
            else:
                mape_actual_rows.append({"region": region, "n_points": 0,
                                         "mape_actual (%)": None})
        st.dataframe(pd.DataFrame(mape_actual_rows), use_container_width=True)


# ----------------------------------------------------------------
# TAB 4 — MULTIVARIATE FORECAST
# ----------------------------------------------------------------
with tab4:
    st.header("Step 4 — Multivariate Forecast (target + exogs)")

    if not exog_cols:
        st.info("ℹ️ Tidak ada kolom EXOG yang dipilih di sidebar. Tab ini hanya aktif untuk multivariate.")
        st.stop()

    if st.session_state.prepared_df is None:
        st.warning("Jalankan Step 1 terlebih dahulu.")
        st.stop()

    prepared_full = st.session_state.prepared_df
    raw_local = st.session_state.raw_df

    st.caption(
        f"**Target:** `{target_col}`  |  **Exogs:** {', '.join(f'`{e}`' for e in exog_cols)}.  "
        "Future exog di setiap region diambil dari hasil Step 3 — Horizon Forecast region tsb."
    )

    # Validasi: tiap region × exog harus punya horizon forecast
    missing_combos = []
    for region in regions_list:
        for ex in exog_cols:
            if (region not in st.session_state.uni_horizon_results
                    or ex not in st.session_state.uni_horizon_results[region]):
                missing_combos.append((region, ex))
    if missing_combos:
        st.error(
            "Kombinasi region × exog berikut belum punya **Horizon Forecast** (Step 3.B):\n\n"
            + "\n".join(f"- region=`{r}`, exog=`{e}`" for r, e in missing_combos)
            + "\n\nKembali ke Step 3, jalankan **Run Horizon Forecast** utk tiap exog (akan loop semua region)."
        )
        st.stop()

    # Validasi: rentang horizon harus konsisten per region (tiap exog di region yg sama harus sama)
    bad_regions = []
    region_horizon = {}
    for region in regions_list:
        starts = set(st.session_state.uni_horizon_params[region][e]["start"] for e in exog_cols)
        ends   = set(st.session_state.uni_horizon_params[region][e]["end"]   for e in exog_cols)
        if len(starts) > 1 or len(ends) > 1:
            bad_regions.append(region)
        else:
            region_horizon[region] = dict(start=starts.pop(), end=ends.pop())

    if bad_regions:
        st.error(
            f"Region berikut punya rentang horizon **tidak konsisten** antar exog: "
            + ", ".join(f"`{r}`" for r in bad_regions)
            + ". Jalankan ulang Step 3.B utk tiap exog dengan start & end yang sama."
        )
        st.stop()

    # Validasi: semua region harus pakai rentang horizon yg sama (utk multivariate consistency)
    all_starts = set(p["start"] for p in region_horizon.values())
    all_ends   = set(p["end"]   for p in region_horizon.values())
    if len(all_starts) > 1 or len(all_ends) > 1:
        st.error(
            "Rentang horizon antar **region berbeda**:\n\n"
            + "\n".join(f"- `{r}`: {p['start']} → {p['end']}" for r, p in region_horizon.items())
            + "\n\nJalankan ulang Step 3.B untuk semua region dengan rentang yang sama."
        )
        st.stop()

    h_start_str = all_starts.pop()
    h_end_str   = all_ends.pop()
    st.success(f"✅ Semua region & exog memakai horizon yg sama: **{h_start_str} → {h_end_str}**.")

    focus_region_t4 = region_focus_selector("focus_region_t4")

    # === 4.A MAPE MODEL ===
    st.subheader(f"4.A MAPE Model — `{target_col}` ~ {' + '.join(exog_cols)}")
    p_min, p_max = prepared_full["ds"].min().date(), prepared_full["ds"].max().date()
    with st.form(key="form_mape_multi"):
        with st.expander("⚙️ Parameter MAPE Model", expanded=True):
            c1, c2, c3 = st.columns(3)
            m_train = c1.date_input("Training cutoff",
                                    value=p_max - pd.Timedelta(days=90),
                                    min_value=p_min, max_value=p_max, key="m_tr")
            m_start = c2.date_input("Start forecast",
                                    value=p_max - pd.Timedelta(days=89),
                                    min_value=p_min, max_value=p_max, key="m_st")
            m_end   = c3.date_input("End forecast",
                                    value=p_max,
                                    min_value=p_min, max_value=p_max, key="m_en")
            params_me = prophet_param_block("me", default_cpr=0.8, default_cps=0.05)
        submitted_mape_multi = st.form_submit_button("🚀 Run MAPE Model (Multivariate)")

    if submitted_mape_multi:
        rows, plots_multi = [], {}
        progress = st.progress(0.0, text="Multivariate forecasting per region...")
        for i, region in enumerate(regions_list):
            sub_prep = slice_region(prepared_full, region)
            if sub_prep.empty:
                continue
            try:
                exec_df, mape_val = run_eval_multi(
                    sub_prep, target_col, exog_cols, holidays_df,
                    params_me["smode"], params_me["yearly"], params_me["weekly"], params_me["daily"],
                    float(params_me["cpr"]), int(params_me["ncp"]), float(params_me["cps"]),
                    float(params_me["spc"]), float(params_me["hps"]),
                    str(m_train), str(m_start), str(m_end),
                )
                rows.append({"region": region, "mape (%)": round(mape_val, 3) if mape_val else None})
                plots_multi[region] = exec_df
            except Exception as ex:
                rows.append({"region": region, "mape (%)": f"error: {ex}"})
            progress.progress((i + 1) / len(regions_list), text=f"Region {region} selesai")
        progress.empty()

        st.subheader("Hasil MAPE Model per Region")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if focus_region_t4 in plots_multi:
            exec_df = plots_multi[focus_region_t4]
            df_fc = exec_df[exec_df["ds"] >= pd.to_datetime(m_start)]
            df_ac = exec_df[exec_df["ds"] <= pd.to_datetime(m_train)][["ds", "y"]].dropna()
            mv = next((r["mape (%)"] for r in rows if r["region"] == focus_region_t4), None)
            title = f"Multivariate MAPE Model — {target_col} · {focus_region_t4}" + (f" | MAPE = {mv} %" if mv else "")
            st.plotly_chart(plot_forecast(df_fc, df_ac, title=title), use_container_width=True)

        if is_multi_region:
            with st.expander("Plot multivariate per region (semua)"):
                for region, exec_df in plots_multi.items():
                    df_fc = exec_df[exec_df["ds"] >= pd.to_datetime(m_start)]
                    df_ac = exec_df[exec_df["ds"] <= pd.to_datetime(m_train)][["ds", "y"]].dropna()
                    mv = next((r["mape (%)"] for r in rows if r["region"] == region), None)
                    st.markdown(f"**Region: `{region}`** — MAPE = {mv}")
                    st.plotly_chart(
                        plot_forecast(df_fc, df_ac, title=f"{target_col} · {region}"),
                        use_container_width=True,
                    )

    st.markdown("---")

    # === 4.B HORIZON ===
    st.subheader(f"4.B Horizon Forecast — `{target_col}` (horizon {h_start_str} → {h_end_str})")
    with st.form(key="form_horizon_multi"):
        with st.expander("⚙️ Parameter Horizon Forecast", expanded=True):
            params_mh = prophet_param_block("mh", default_cpr=0.8, default_cps=0.05)
        submitted_horizon_multi = st.form_submit_button("🚀 Run Horizon Forecast (Multivariate)", type="primary")

    if submitted_horizon_multi:
        progress = st.progress(0.0, text="Multivariate horizon forecasting per region...")
        for i, region in enumerate(regions_list):
            sub_prep = slice_region(prepared_full, region)
            if sub_prep.empty:
                continue
            # Build future_exog_df khusus region ini
            horizon_dates = pd.date_range(h_start_str, h_end_str, freq="D")
            future_exog_df = pd.DataFrame({"ds": horizon_dates})
            for ex in exog_cols:
                uni_hr = st.session_state.uni_horizon_results[region][ex]
                ex_part = uni_hr.loc[uni_hr["ds"].isin(horizon_dates),
                                     ["ds", "yhat"]].rename(columns={"yhat": ex})
                future_exog_df = future_exog_df.merge(ex_part, on="ds", how="left")
            try:
                multi_horizon = run_horizon_multi(
                    sub_prep, target_col, exog_cols, future_exog_df, holidays_df,
                    params_mh["smode"], params_mh["yearly"], params_mh["weekly"], params_mh["daily"],
                    float(params_mh["cpr"]), int(params_mh["ncp"]), float(params_mh["cps"]),
                    float(params_mh["spc"]), float(params_mh["hps"]),
                    h_start_str, h_end_str,
                )
                st.session_state.multi_horizon_results[region] = multi_horizon
                st.session_state.multi_horizon_params[region]  = dict(start=h_start_str, end=h_end_str)
            except Exception as ex:
                st.error(f"Region `{region}` gagal: {ex}")
            progress.progress((i + 1) / len(regions_list), text=f"Region {region} selesai")
        progress.empty()
        st.success(f"✅ Multivariate horizon tersimpan utk {len(st.session_state.multi_horizon_results)} region.")

    # Display: focused region
    if focus_region_t4 in st.session_state.multi_horizon_results:
        mh = st.session_state.multi_horizon_results[focus_region_t4]
        params = st.session_state.multi_horizon_params[focus_region_t4]
        df_fc = mh[mh["ds"] >= pd.to_datetime(params["start"])]
        df_ac = mh[mh["ds"] < pd.to_datetime(params["start"])][["ds", "y"]].dropna()
        st.plotly_chart(
            plot_forecast(df_fc, df_ac,
                          title=f"Multivariate Horizon — {target_col} · {focus_region_t4} ({params['start']} → {params['end']})"),
            use_container_width=True,
        )

        # === 4.C MAPE ACTUAL ===
        st.markdown("---")
        st.subheader(f"4.C MAPE Actual — `{target_col}` (multivariate)")
        mape_rows = []
        for region in regions_list:
            if region not in st.session_state.multi_horizon_results:
                continue
            mh_r = st.session_state.multi_horizon_results[region]
            pr   = st.session_state.multi_horizon_params[region]
            forecast_h = mh_r.loc[
                mh_r["ds"].between(pr["start"], pr["end"]),
                ["ds", "yhat"]
            ].rename(columns={"yhat": "forecast"})
            raw_r = slice_region(raw_local, region)
            actual_h = raw_r.loc[
                raw_r["ds"].between(pr["start"], pr["end"]),
                ["ds", target_col]
            ].rename(columns={target_col: "actual"})
            eval_h = pd.merge(actual_h, forecast_h, on="ds", how="inner").dropna()
            if len(eval_h):
                mape_actual = mean_absolute_percentage_error(eval_h["actual"], eval_h["forecast"]) * 100
                mape_rows.append({"region": region, "n_points": len(eval_h),
                                  "mape_actual (%)": round(mape_actual, 3)})
            else:
                mape_rows.append({"region": region, "n_points": 0,
                                  "mape_actual (%)": None})
        st.dataframe(pd.DataFrame(mape_rows), use_container_width=True)

        # Download — gabungan semua region
        all_horizons = []
        for region, df in st.session_state.multi_horizon_results.items():
            df_copy = df.copy()
            df_copy["region"] = region
            all_horizons.append(df_copy)
        combined = pd.concat(all_horizons, ignore_index=True)
        csv_buf = io.StringIO()
        combined.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Multivariate Forecast — ALL regions (CSV)",
            csv_buf.getvalue(),
            file_name=f"multivariate_forecast_{target_col}.csv",
            mime="text/csv",
        )
