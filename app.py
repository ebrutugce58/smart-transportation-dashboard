"""
Smart Bus Arrival Prediction — CSV-driven: arrivals, weather observations, trips, passenger flow.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, make_response, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
BUS_LINES = ("L01", "L02", "L03", "L04", "L05")

FALLBACK_STOPS: tuple[tuple[str, str], ...] = (
    ("S1", "Downtown Hub"),
    ("S2", "Market Street"),
    ("S3", "University Gate"),
    ("S4", "Riverside Park"),
    ("S5", "Airport Connector"),
)

_arrivals_df: pd.DataFrame | None = None
_weather_df: pd.DataFrame | None = None
_trips_df: pd.DataFrame | None = None
_flow_df: pd.DataFrame | None = None
_bus_stops_df: pd.DataFrame | None = None
_load_note: str | None = None
_rf_artifact: dict[str, Any] | None = None


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _init_data() -> None:
    global _arrivals_df, _weather_df, _trips_df, _flow_df, _bus_stops_df, _load_note
    notes: list[str] = []

    arr = _safe_read_csv(BASE_DIR / "stop_arrivals.csv")
    if arr is None or arr.empty:
        notes.append("stop_arrivals.csv unavailable.")
        arr = None
    else:
        required_a = {"line_id", "stop_id", "minutes_to_next_bus", "delay_min", "hour_of_day"}
        if not required_a.issubset(set(arr.columns)):
            notes.append("stop_arrivals.csv missing required columns.")
            arr = None

    wx = _safe_read_csv(BASE_DIR / "weather_observations.csv")
    if wx is None or wx.empty:
        notes.append("weather_observations.csv unavailable.")
        wx = None
    elif "timestamp" not in wx.columns or "weather_condition" not in wx.columns:
        notes.append("weather_observations.csv missing timestamp or weather_condition.")
        wx = None

    trips = _safe_read_csv(BASE_DIR / "bus_trips.csv")
    if trips is None or trips.empty:
        notes.append("bus_trips.csv unavailable.")
        trips = None
    elif not {"line_id", "planned_departure", "date"}.issubset(set(trips.columns)):
        notes.append("bus_trips.csv missing required columns.")
        trips = None

    flow = _safe_read_csv(BASE_DIR / "passenger_flow.csv")
    if flow is None or flow.empty:
        notes.append("passenger_flow.csv unavailable.")
        flow = None
    elif not {"line_id", "stop_id", "hour_of_day", "avg_passengers_waiting"}.issubset(
        set(flow.columns)
    ):
        notes.append("passenger_flow.csv missing required columns.")
        flow = None

    bst = _safe_read_csv(BASE_DIR / "bus_stops.csv")
    if bst is None or bst.empty:
        bst = None
    elif not {"line_id", "stop_sequence", "stop_type"}.issubset(set(bst.columns)):
        bst = None

    _arrivals_df = arr
    _weather_df = wx
    _trips_df = trips
    _flow_df = flow
    _bus_stops_df = bst
    _load_note = " ".join(notes) if notes else None


def _line_name_from_bus_stops(line_id: str) -> str | None:
    """Exact line_name from bus_stops.csv (dataset label)."""
    if _bus_stops_df is None or _bus_stops_df.empty:
        return None
    if "line_name" not in _bus_stops_df.columns:
        return None
    try:
        sub = _bus_stops_df[_bus_stops_df["line_id"] == line_id]
        if sub.empty:
            return None
        v = sub.iloc[0].get("line_name")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        return s or None
    except Exception:
        return None


def _stop_type_display(raw: str) -> str:
    """English UI labels for CSV stop_type (e.g. terminal → Terminal, university → University)."""
    s = str(raw or "").strip()
    if not s:
        return ""
    return s.replace("_", " ").title()


def _stop_type_label_for_line_stop(line_id: str, stop_id: str, arrivals_row: pd.Series | None) -> str:
    """stop_type from bus_stops or arrivals, capitalized for display."""
    if _bus_stops_df is not None and not _bus_stops_df.empty:
        try:
            sub = _bus_stops_df[
                (_bus_stops_df["line_id"] == line_id) & (_bus_stops_df["stop_id"] == stop_id)
            ]
            if not sub.empty:
                v = sub.iloc[0].get("stop_type")
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    s = str(v).strip()
                    if s:
                        return _stop_type_display(s)
        except Exception:
            pass
    if arrivals_row is not None:
        v = arrivals_row.get("stop_type")
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            s = str(v).strip()
            if s:
                return _stop_type_display(s)
    return stop_id


def _format_stop_dropdown_label(row: pd.Series) -> str:
    """Readable label: capitalized stop_type, then Stop N; value attribute holds stop_id."""
    stype = str(row.get("stop_type", "") or "").strip()
    sid = str(row.get("stop_id", "") or "").strip()
    stype_disp = _stop_type_display(stype)
    seq_raw = row.get("stop_sequence")
    seq_i: int | None = None
    try:
        if seq_raw is not None and not (isinstance(seq_raw, float) and pd.isna(seq_raw)):
            seq_i = int(seq_raw)
    except (TypeError, ValueError):
        seq_i = None
    if stype and seq_i is not None:
        return f"{stype_disp} · Stop {seq_i}"
    if stype:
        return f"{stype_disp} · {sid}"
    return sid


def _stops_by_line_from_bus_stops(bst: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for line in BUS_LINES:
        sub = bst[bst["line_id"] == line].sort_values("stop_sequence")
        rows: list[tuple[str, str]] = []
        for _, row in sub.iterrows():
            sid = str(row["stop_id"])
            rows.append((sid, _format_stop_dropdown_label(row)))
        out[line] = rows
    return out


def _stops_by_line_from_arrivals(df: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for line in BUS_LINES:
        sub = (
            df[df["line_id"] == line]
            .sort_values(["stop_sequence", "stop_id"])
            .drop_duplicates(subset=["stop_id"], keep="first")
        )
        rows: list[tuple[str, str]] = []
        for _, row in sub.iterrows():
            sid = str(row["stop_id"])
            rows.append((sid, _format_stop_dropdown_label(row)))
        out[line] = rows
    return out


def _fallback_stops_by_line() -> dict[str, list[tuple[str, str]]]:
    return {line: list(FALLBACK_STOPS) for line in BUS_LINES}


def stops_by_line() -> dict[str, list[tuple[str, str]]]:
    if _bus_stops_df is not None and not _bus_stops_df.empty:
        return _stops_by_line_from_bus_stops(_bus_stops_df)
    if _arrivals_df is not None:
        return _stops_by_line_from_arrivals(_arrivals_df)
    return _fallback_stops_by_line()


def line_options() -> list[tuple[str, str]]:
    """(line_id, option label): CSV line_name first, then id in parentheses."""
    out: list[tuple[str, str]] = []
    for line_id in BUS_LINES:
        nm = _line_name_from_bus_stops(line_id)
        if nm:
            out.append((line_id, f"{nm} ({line_id})"))
        else:
            out.append((line_id, line_id))
    return out


def route_preview_by_line() -> dict[str, str]:
    """Condensed route path string per line_id for UI preview."""
    return {lid: (_line_route_preview(lid) or "—") for lid in BUS_LINES}


def _line_route_preview(line_id: str, max_segments: int = 64) -> str:
    """Full route shape from bus_stops.csv (ordered stop_sequence), English title case."""
    if _bus_stops_df is None or _bus_stops_df.empty:
        return ""
    try:
        sub = _bus_stops_df[_bus_stops_df["line_id"] == line_id].sort_values("stop_sequence")
        parts: list[str] = []
        prev_raw: str | None = None
        for _, row in sub.iterrows():
            raw = str(row.get("stop_type", "") or "").strip()
            if not raw:
                continue
            if raw != prev_raw:
                parts.append(_stop_type_display(raw))
                prev_raw = raw
            if len(parts) >= max_segments:
                break
        return " → ".join(parts) if parts else ""
    except Exception:
        return ""


def _bus_stop_row(line_id: str, stop_id: str) -> pd.Series | None:
    if _bus_stops_df is None or _bus_stops_df.empty:
        return None
    try:
        sub = _bus_stops_df[
            (_bus_stops_df["line_id"] == line_id) & (_bus_stops_df["stop_id"] == stop_id)
        ]
        return sub.iloc[0] if not sub.empty else None
    except Exception:
        return None


def _is_transfer_hub_stop(line_id: str, stop_id: str) -> bool:
    row = _bus_stop_row(line_id, stop_id)
    if row is None:
        return False
    try:
        v = row.get("is_transfer_hub")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return int(v) == 1
    except (TypeError, ValueError):
        return False


def _shelter_available(line_id: str, stop_id: str) -> bool:
    row = _bus_stop_row(line_id, stop_id)
    if row is None:
        return False
    try:
        v = row.get("shelter_available")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return int(v) == 1
    except (TypeError, ValueError):
        return False


def _normalize_traffic_arrivals(val: str) -> str | None:
    r = (val or "").strip().lower()
    if r == "low":
        return "Low"
    if r == "moderate":
        return "Moderate"
    if r in ("high", "congested"):
        return "Heavy"
    return None


def _traffic_level_from_arrivals(use: pd.DataFrame) -> str:
    """Primary: mode of traffic_level in stop_arrivals.csv; else infer from delays."""
    if "traffic_level" not in use.columns or use.empty:
        t, _ = _infer_traffic_from_delay_patterns(use)
        return t
    raw = use["traffic_level"].dropna().astype(str)
    if raw.empty:
        t, _ = _infer_traffic_from_delay_patterns(use)
        return t
    mapped = [_normalize_traffic_arrivals(x) for x in raw.tolist()]
    mapped = [m for m in mapped if m]
    if not mapped:
        t, _ = _infer_traffic_from_delay_patterns(use)
        return t
    ser = pd.Series(mapped)
    try:
        mode = ser.mode().iloc[0] if len(ser.mode()) else ser.iloc[0]
        return str(mode)
    except Exception:
        t, _ = _infer_traffic_from_delay_patterns(use)
        return t


def _traffic_level_from_trips(line_id: str, hour: int) -> str | None:
    """Mode of bus_trips.traffic_level for the line (hour slice preferred)."""
    if _trips_df is None or _trips_df.empty or "traffic_level" not in _trips_df.columns:
        return None
    try:
        t = _trips_df[_trips_df["line_id"] == line_id].copy()
        t["pd"] = pd.to_datetime(t["planned_departure"], errors="coerce")
        t = t.dropna(subset=["pd"])
        if t.empty:
            return None
        sub = t[t["pd"].dt.hour == hour]
        if len(sub) < 8:
            sub = t
        raw = sub["traffic_level"].dropna().astype(str)
        if raw.empty:
            return None
        mapped = [_normalize_traffic_arrivals(x) for x in raw.tolist()]
        mapped = [m for m in mapped if m]
        if not mapped:
            return None
        ser = pd.Series(mapped)
        return str(ser.mode().iloc[0] if len(ser.mode()) else ser.iloc[0])
    except Exception:
        return None


def _weather_from_arrivals_slice(use: pd.DataFrame) -> str | None:
    """Mode of weather_condition on a stop_arrivals slice (raw CSV values)."""
    if use.empty or "weather_condition" not in use.columns:
        return None
    raw = _mode_series_str(use["weather_condition"].astype(str), "")
    if not raw or raw.lower() in ("nan", "none"):
        return None
    return _map_weather_display(raw)


def _select_arrivals_use(
    df: pd.DataFrame, line: str, stop_id: str, hour: int
) -> tuple[pd.DataFrame, str]:
    """
    Rows from stop_arrivals for ETA/traffic/confidence. Prefer same stop + hour;
    else all hours for that stop; else all observations on the line.
    Returns (use_df, scope_tag for lineage).
    """
    sub = df[(df["line_id"] == line) & (df["stop_id"] == stop_id)]
    if sub.empty:
        line_df = df[df["line_id"] == line]
        if line_df.empty:
            return pd.DataFrame(), "none"
        return line_df.copy(), "stop_arrivals.csv (line-level; no rows for this stop)"
    hour_sub = sub[sub["hour_of_day"] == hour]
    if len(hour_sub) >= 5:
        return hour_sub.copy(), "stop_arrivals.csv (same stop + hour_of_day)"
    return sub.copy(), "stop_arrivals.csv (same stop; all hours — fewer than 5 at this hour)"


def _is_rush_hour(hour: int) -> bool:
    return (7 <= hour <= 9) or (16 <= hour <= 19)


def _map_weather_display(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r == "clear":
        return "Clear"
    if r == "cloudy":
        return "Cloudy"
    if r == "rain":
        return "Rain"
    if r == "wind":
        return "Windy"
    if r == "snow":
        return "Snow"
    if r in ("fog",):
        return "Cloudy"
    return "Cloudy"


def _weather_eta_from_observation(row: pd.Series | None) -> tuple[float, str]:
    """Extra minutes from weather_observations row (transit_delay_risk + precipitation)."""
    if row is None:
        return 0.0, "No weather row selected."
    try:
        risk = float(row.get("transit_delay_risk", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk = 0.0
    try:
        precip = float(row.get("precipitation_mm", 0.0) or 0.0)
    except (TypeError, ValueError):
        precip = 0.0
    extra = 12.0 * max(0.0, min(risk, 0.95)) + min(4.0, precip * 0.12)
    detail = f"About +{extra:.1f} min applied for modeled conditions."
    return float(extra), detail


def _resolve_weather_observation(
    now: datetime, line_id: str, stop_id: str
) -> tuple[pd.Series | None, int, int]:
    """
    Pick a deterministic weather_observations row (same hour; prefer same month).
    Returns (row, pool size, index) or (None, 0, -1).
    """
    if _weather_df is None or _weather_df.empty:
        return None, 0, -1
    try:
        wx = _weather_df.copy()
        ts = pd.to_datetime(wx["timestamp"], errors="coerce")
        wx = wx.loc[ts.notna()].copy()
        wx["_ts"] = pd.to_datetime(wx["timestamp"], errors="coerce")
        wx["_h"] = wx["_ts"].dt.hour
        target_h = int(now.hour)
        target_m = int(now.month)
        same_hour = wx[wx["_h"] == target_h]
        same_month = same_hour[same_hour["_ts"].dt.month == target_m]
        if len(same_month) >= 2:
            pool = same_month
        elif len(same_hour) >= 2:
            pool = same_hour
        else:
            pool = wx
        n = len(pool)
        if n == 0:
            return None, 0, -1
        pool = pool.sort_values("_ts").reset_index(drop=True)
        key = f"{line_id}|{stop_id}|{now.date().isoformat()}|{target_h}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        idx = int(digest[:16], 16) % n
        rep = pool.iloc[idx]
        return rep, n, idx
    except Exception:
        return None, 0, -1


def _infer_traffic_from_delay_patterns(use: pd.DataFrame) -> tuple[str, str]:
    """
    Traffic from delay_min distribution for this line/stop/hour slice (stop_arrivals).
    Returns (Low|Moderate|Heavy, short reason).
    """
    if use.empty or "delay_min" not in use.columns:
        return "Moderate", "Insufficient delay samples."
    d = pd.to_numeric(use["delay_min"], errors="coerce").dropna()
    if d.empty:
        return "Moderate", "No valid delay_min values."
    m = float(d.mean())
    p75 = float(d.quantile(0.75))
    p90 = float(d.quantile(0.90))
    late = float((d > 2.5).mean())
    if m >= 7.0 or p90 >= 14.0 or late >= 0.52:
        return "Heavy", f"mean delay {m:.1f} min, p90 {p90:.1f} min, {late:.0%} trips >2.5 min late."
    if m >= 3.0 or p75 >= 6.5 or late >= 0.28:
        return "Moderate", f"mean delay {m:.1f} min, p75 {p75:.1f} min, {late:.0%} late fraction."
    return "Low", f"mean delay {m:.1f} min, p75 {p75:.1f} min — relatively on-time."


def _traffic_eta_adjustment(traffic: str) -> float:
    return {"Low": -1.0, "Moderate": 0.0, "Heavy": 2.5}.get(traffic, 0.0)


def _median_planned_headway_minutes(line_id: str, hour: int | None) -> float | None:
    if _trips_df is None or _trips_df.empty:
        return None
    try:
        t = _trips_df[_trips_df["line_id"] == line_id].copy()
        t["pd"] = pd.to_datetime(t["planned_departure"], errors="coerce")
        t = t.dropna(subset=["pd"])
        if hour is not None:
            t = t[t["pd"].dt.hour == hour]
        if t.empty:
            return None
        gaps: list[float] = []
        for _, grp in t.groupby("date"):
            grp = grp.sort_values("pd")
            ts = grp["pd"].to_numpy()
            for i in range(1, len(ts)):
                delta_min = (pd.Timestamp(ts[i]) - pd.Timestamp(ts[i - 1])).total_seconds() / 60.0
                if 1.0 <= delta_min <= 180.0:
                    gaps.append(delta_min)
        if not gaps:
            return None
        return float(pd.Series(gaps).median())
    except Exception:
        return None


def _scheduled_eta_from_trips(line_id: str, hour: int) -> float | None:
    hw = _median_planned_headway_minutes(line_id, hour)
    if hw is not None:
        return hw
    return _median_planned_headway_minutes(line_id, None)


def _minutes_offset_to_stop(line_id: str, stop_id: str) -> float:
    """Cumulative scheduled travel minutes from route start to this stop (bus_stops.csv)."""
    if _bus_stops_df is None or _bus_stops_df.empty:
        return 0.0
    try:
        b = _bus_stops_df[_bus_stops_df["line_id"] == line_id].sort_values("stop_sequence")
        row = b[b["stop_id"] == stop_id]
        if row.empty:
            return 0.0
        max_seq = int(row.iloc[0]["stop_sequence"])
        sub = b[b["stop_sequence"] <= max_seq]
        return float(pd.to_numeric(sub["scheduled_travel_time_min"], errors="coerce").fillna(0.0).sum())
    except Exception:
        return 0.0


def _format_clock_12h(dt: datetime) -> str:
    h24 = dt.hour
    h12 = h24 % 12
    if h12 == 0:
        h12 = 12
    ap = "AM" if h24 < 12 else "PM"
    return f"{h12}:{dt.minute:02d} {ap}"


def _next_arrivals_table(
    line_id: str,
    stop_id: str,
    now: datetime,
    eta_minutes: int,
    headway_min: float | None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Next 3 arrival clock times at this stop: bus_trips departure times + offset from bus_stops,
    projected onto today/tomorrow; gaps filled with headway. Primary row aligns with predicted ETA.
    """
    primary = now + timedelta(minutes=float(eta_minutes))
    hw = float(headway_min if headway_min is not None else 12.0)
    if hw < 1.0:
        hw = 12.0

    offset = _minutes_offset_to_stop(line_id, stop_id)
    candidates: list[datetime] = []
    source = "Derived from ETA and scheduled headway (no trip times for this line)"

    if _trips_df is not None and not _trips_df.empty:
        try:
            t = _trips_df[_trips_df["line_id"] == line_id].copy()
            t["pd"] = pd.to_datetime(t["planned_departure"], errors="coerce")
            t = t.dropna(subset=["pd"])
            uniq_hm: set[tuple[int, int]] = set()
            for ts in t["pd"]:
                dep = pd.Timestamp(ts)
                arr = dep + pd.Timedelta(minutes=offset)
                uniq_hm.add((int(arr.hour), int(arr.minute)))
            for day_add in range(0, 4):
                d = (now + timedelta(days=day_add)).date()
                for h, m in sorted(uniq_hm):
                    dt = datetime(d.year, d.month, d.day, h, m, 0)
                    if dt > now:
                        candidates.append(dt)
            candidates = sorted(set(candidates))
            if len(candidates) > 0:
                source = "bus_trips.csv + bus_stops.csv (planned departures + travel to stop)"
        except Exception:
            candidates = []

    times: list[datetime] = [primary]
    for c in candidates:
        if len(times) >= 3:
            break
        if c > primary + timedelta(seconds=5):
            times.append(c)
    while len(times) < 3:
        times.append(times[-1] + timedelta(minutes=hw))

    rows: list[dict[str, Any]] = []
    for a in times[:3]:
        ms = int(a.timestamp() * 1000)
        mins_away = max(0, int(math.ceil((a - now).total_seconds() / 60.0)))
        rows.append(
            {
                "epoch_ms": ms,
                "clock": _format_clock_12h(a),
                "minutes_away": mins_away,
            }
        )

    return rows, source


def _countdown_fields(now: datetime, eta_minutes: int) -> tuple[int, int]:
    """predicted_at and eta_target wall-clock (ms since epoch, local naive)."""
    predicted_at_ms = int(now.timestamp() * 1000)
    target = now + timedelta(minutes=float(eta_minutes))
    eta_target_ms = int(target.timestamp() * 1000)
    return predicted_at_ms, eta_target_ms


def _route_stop_count(line_id: str) -> int | None:
    if _bus_stops_df is not None and not _bus_stops_df.empty:
        try:
            n = len(_bus_stops_df[_bus_stops_df["line_id"] == line_id])
            if n:
                return int(n)
        except Exception:
            pass
    if _arrivals_df is not None and not _arrivals_df.empty:
        try:
            sub = _arrivals_df[_arrivals_df["line_id"] == line_id]
            nu = sub["stop_id"].nunique() if "stop_id" in sub.columns else 0
            if nu:
                return int(nu)
        except Exception:
            pass
    return None


def _route_topology_label(line_id: str) -> str:
    """Short route shape for UI (e.g. bidirectional vs loop)."""
    if _bus_stops_df is None or _bus_stops_df.empty:
        return "Route"
    try:
        sub = _bus_stops_df[_bus_stops_df["line_id"] == line_id].sort_values("stop_sequence")
        if sub.empty:
            return "Route"
        first = str(sub.iloc[0]["stop_id"])
        last = str(sub.iloc[-1]["stop_id"])
        if first == last:
            return "Circular route"
        return "Bidirectional route"
    except Exception:
        return "Route"


def _stop_median_minutes_to_next(line_id: str, stop_id: str) -> float | None:
    if _arrivals_df is None or _arrivals_df.empty:
        return None
    try:
        sub = _arrivals_df[
            (_arrivals_df["line_id"] == line_id) & (_arrivals_df["stop_id"] == stop_id)
        ]
        if sub.empty or "minutes_to_next_bus" not in sub.columns:
            return None
        s = pd.to_numeric(sub["minutes_to_next_bus"], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.median())
    except Exception:
        return None


def _confidence_reliability_label(pct: int) -> str:
    if pct < 70:
        return "Low"
    if pct < 84:
        return "Medium"
    return "High"


def _enrich_transit_ui(pred: dict[str, Any], line: str, stop_id: str) -> dict[str, Any]:
    """Countdown anchor + next 3 arrivals for the dashboard (all prediction paths)."""
    now = datetime.now()
    eta_i = int(pred.get("eta_minutes", 12))
    sched = pred.get("scheduled_eta_minutes")
    try:
        hw_f = float(sched) if sched is not None else 12.0
    except (TypeError, ValueError):
        hw_f = 12.0
    pred_ms, eta_ms = _countdown_fields(now, eta_i)
    rows, src = _next_arrivals_table(line, stop_id, now, eta_i, hw_f)
    out = dict(pred)
    out["predicted_at_ms"] = pred_ms
    out["eta_target_ms"] = eta_ms
    out["next_arrivals"] = rows
    out["next_arrivals_source"] = src

    n_stops = _route_stop_count(line)
    out["route_stop_count"] = n_stops
    out["route_topology_label"] = _route_topology_label(line)

    hist_med = _stop_median_minutes_to_next(line, stop_id)
    try:
        sched_i = int(round(float(sched))) if sched is not None else int(round(hw_f))
    except (TypeError, ValueError):
        sched_i = 12
    if hist_med is not None and not math.isnan(hist_med):
        out["stop_avg_eta_minutes"] = int(round(max(2.0, min(90.0, hist_med))))
        out["stop_avg_eta_source"] = "history"
    else:
        out["stop_avg_eta_minutes"] = max(2, min(90, sched_i))
        out["stop_avg_eta_source"] = "schedule"

    conf = int(out.get("confidence_pct", 70))
    out["confidence_reliability"] = _confidence_reliability_label(conf)

    return out


def _fallback_headway_from_trip_duration(line_id: str) -> float | None:
    """Rough segment time from bus_trips planned_duration_min / num_stops when headway gaps are empty."""
    if _trips_df is None or _trips_df.empty:
        return None
    if "planned_duration_min" not in _trips_df.columns:
        return None
    try:
        t = _trips_df[_trips_df["line_id"] == line_id]
        if t.empty:
            return None
        dur = pd.to_numeric(t["planned_duration_min"], errors="coerce").dropna()
        if dur.empty:
            return None
        med_d = float(dur.median())
        if "num_stops" in t.columns:
            ns = pd.to_numeric(t["num_stops"], errors="coerce").dropna()
            n_med = float(ns.median()) if not ns.empty else 12.0
        else:
            n_med = 12.0
        n_med = max(4.0, min(40.0, n_med))
        est = med_d / n_med
        return max(3.0, min(45.0, float(est)))
    except Exception:
        return None


def _matching_passenger_flow_rows(line_id: str, stop_id: str, now: datetime) -> pd.DataFrame:
    if _flow_df is None or _flow_df.empty:
        return pd.DataFrame()
    try:
        pf = _flow_df
        dow = int(now.weekday())
        wknd = 1 if dow >= 5 else 0
        h = int(now.hour)
        sub = pf[
            (pf["line_id"] == line_id)
            & (pf["stop_id"] == stop_id)
            & (pf["hour_of_day"] == h)
            & (pf["day_of_week"] == dow)
            & (pf["is_weekend"] == wknd)
        ]
        if len(sub) < 1:
            sub = pf[(pf["line_id"] == line_id) & (pf["stop_id"] == stop_id) & (pf["hour_of_day"] == h)]
        if len(sub) < 1:
            sub = pf[(pf["line_id"] == line_id) & (pf["stop_id"] == stop_id)]
        return sub.copy()
    except Exception:
        return pd.DataFrame()


def _flow_sample_and_waiting_std(sub: pd.DataFrame) -> tuple[float, float | None]:
    """Total sample_count and weighted mean of std_passengers_waiting for confidence."""
    if sub.empty:
        return 0.0, None
    try:
        if "sample_count" in sub.columns:
            w = pd.to_numeric(sub["sample_count"], errors="coerce").fillna(1.0)
            sc = float(w.sum())
        else:
            w = pd.Series([1.0] * len(sub))
            sc = float(len(sub))
        if "std_passengers_waiting" in sub.columns:
            stds = pd.to_numeric(sub["std_passengers_waiting"], errors="coerce")
            if stds.notna().any():
                agg = float((stds.fillna(0.0) * w).sum() / max(float(w.sum()), 1e-9))
                return sc, agg
        return sc, None
    except Exception:
        return 0.0, None


def _line_occupancy_tier(line_id: str, hour: int) -> str | None:
    """low | moderate | high from bus_trips avg_occupancy_pct (same hour preferred)."""
    if _trips_df is None or _trips_df.empty or "avg_occupancy_pct" not in _trips_df.columns:
        return None
    try:
        t = _trips_df[_trips_df["line_id"] == line_id].copy()
        t["pd"] = pd.to_datetime(t["planned_departure"], errors="coerce")
        t = t.dropna(subset=["pd"])
        if t.empty:
            return None
        sub = t[t["pd"].dt.hour == hour]
        if len(sub) < 5:
            sub = t
        occ = pd.to_numeric(sub["avg_occupancy_pct"], errors="coerce").dropna()
        if occ.empty:
            return None
        med = float(occ.median())
        if med >= 72.0:
            return "high"
        if med >= 45.0:
            return "moderate"
        return "low"
    except Exception:
        return None


def _crowding_display_and_badge(
    line_id: str, stop_id: str, now: datetime, flow_sub: pd.DataFrame
) -> tuple[str | None, str]:
    """
    Human-readable demand line (no raw counts) and badge key dem-low|dem-mod|dem-high.
    """
    hour = int(now.hour)
    occ = _line_occupancy_tier(line_id, hour)

    flow_mode: str | None = None
    if not flow_sub.empty and "crowding_level" in flow_sub.columns:
        flow_mode = _mode_series_str(flow_sub["crowding_level"].astype(str), "moderate").lower()

    parts: list[str] = []
    if occ == "high":
        parts.append("High crowding risk from typical bus occupancy on this line")
    elif occ == "moderate":
        parts.append("Moderate bus occupancy on this line")
    elif occ == "low":
        parts.append("Lower bus occupancy on this line")

    if flow_mode:
        fl = flow_mode.replace("_", " ").strip().title()
        parts.append(f"Stop platform pattern: {fl}")

    if not parts:
        if flow_sub.empty:
            return None, "dem-mod"
        cl = _mode_series_str(flow_sub["crowding_level"].astype(str), "") if "crowding_level" in flow_sub.columns else ""
        if cl:
            return (
                f"Stop crowding from passenger_flow.csv: {cl.replace('_', ' ').strip().title()}",
                "dem-mod",
            )
        return "passenger_flow.csv rows matched; extend occupancy data for a fuller read", "dem-mod"

    score = 0
    if occ == "high":
        score += 3
    elif occ == "moderate":
        score += 2
    elif occ == "low":
        score += 0
    else:
        score += 1
    if flow_mode == "crowded":
        score += 3
    elif flow_mode == "busy":
        score += 2
    elif flow_mode == "moderate":
        score += 1
    elif flow_mode in ("light", "empty"):
        score -= 1

    if score >= 4:
        badge = "dem-high"
    elif score <= 0:
        badge = "dem-low"
    else:
        badge = "dem-mod"

    return " · ".join(parts), badge


def _confidence_from_real_data(
    use: pd.DataFrame,
    now: datetime,
    traffic: str,
    weather_label: str,
    flow_sample_sum: float = 0.0,
    flow_waiting_std_agg: float | None = None,
) -> int:
    """
    Confidence 55–95 from: arrival row count, delay variance, passenger_flow sample_count
    and spread in waiting-time std, recency, traffic/weather uncertainty.
    If use is empty (no stop_arrivals rows), uses flow + penalties only (capped lower).
    """
    if use is None or len(use) < 1:
        flow_depth = min(7.0, 1.85 * math.log1p(max(flow_sample_sum, 0.0)))
        flow_var_pen = 0.0
        if flow_waiting_std_agg is not None and not math.isnan(flow_waiting_std_agg):
            flow_var_pen = min(5.0, max(0.0, float(flow_waiting_std_agg)) / 28.0)
        t_pen = {"Heavy": 6.0, "Moderate": 2.5, "Low": 0.0}.get(traffic, 2.5)
        w_pen = {"Snow": 5.0, "Rain": 4.0, "Windy": 2.5, "Cloudy": 2.0, "Clear": 0.0}.get(
            weather_label, 2.0
        )
        score = 58.0 + flow_depth - flow_var_pen - t_pen - w_pen
        return int(round(max(55, min(88, score))))

    n = len(use)
    delays = pd.to_numeric(use["delay_min"], errors="coerce").dropna()
    if len(delays) > 1:
        dmean = float(abs(delays.mean()))
        dstd = float(delays.std(ddof=1))
        cv = dstd / max(dmean, 0.65)
    else:
        cv = 2.8

    # Arrival history depth
    depth = min(16.0, 3.0 * math.sqrt(max(n, 1)))

    # Delay consistency (coefficient-like)
    consist = min(11.0, 11.0 / (1.0 + min(cv, 6.0) * 0.82))

    # Passenger-flow aggregate sample size (higher → more trust in demand side)
    flow_depth = min(7.0, 1.85 * math.log1p(max(flow_sample_sum, 0.0)))

    # Higher variance in reported waiting std → slightly lower confidence
    flow_var_pen = 0.0
    if flow_waiting_std_agg is not None and not math.isnan(flow_waiting_std_agg):
        flow_var_pen = min(5.0, max(0.0, float(flow_waiting_std_agg)) / 28.0)

    recency_pts = 6.0
    if "date" in use.columns:
        try:
            dt = pd.to_datetime(use["date"], errors="coerce")
            latest = dt.max()
            if pd.notna(latest):
                days = (pd.Timestamp(now) - latest).total_seconds() / 86400.0
                days = max(0.0, float(days))
                recency_pts = min(12.0, 12.0 * math.exp(-days / 95.0))
        except Exception:
            recency_pts = 6.0

    t_pen = {"Heavy": 6.0, "Moderate": 2.5, "Low": 0.0}.get(traffic, 2.5)
    w_pen = {"Snow": 5.0, "Rain": 4.0, "Windy": 2.5, "Cloudy": 2.0, "Clear": 0.0}.get(
        weather_label, 2.0
    )

    score = (
        55.0
        + depth
        + consist
        + flow_depth
        + recency_pts
        - flow_var_pen
        - t_pen
        - w_pen
    )
    return int(round(max(55, min(95, score))))


def _recommendation_from_comparison(
    scheduled_min: int,
    eta_min: int,
    traffic: str,
    demand_lbl: str,
) -> str:
    """Single-sentence investor-facing line (delta vs timetable)."""
    _ = traffic, demand_lbl  # retained for API compatibility / future use
    delta = int(eta_min - scheduled_min)
    if delta >= 4:
        return (
            f"Predicted wait is ~{eta_min} min versus ~{scheduled_min} min timetable headway "
            f"(~{delta} min over — allow buffer)."
        )
    if delta <= -4:
        return (
            f"Predicted wait is ~{eta_min} min versus ~{scheduled_min} min timetable headway "
            f"(~{abs(delta)} min under nominal spacing)."
        )
    return (
        f"Predicted wait (~{eta_min} min) aligns with scheduled headway (~{scheduled_min} min)."
    )


def _mode_series_str(series: pd.Series, default: str) -> str:
    try:
        m = series.dropna().astype(str).mode()
        if len(m):
            return str(m.iloc[0])
    except Exception:
        pass
    return default


def _predict_trips_flow_weather_only(line: str, stop_id: str) -> dict[str, Any]:
    """
    When stop_arrivals has no rows for this line: ETA/schedule/traffic from bus_trips,
    weather from weather_observations, demand from bus_trips + passenger_flow, confidence from flow + penalties.
    """
    now = datetime.now()
    hour = now.hour
    traffic = _traffic_level_from_trips(line, hour)
    if traffic is None:
        traffic = "Moderate"

    sched_trip = _scheduled_eta_from_trips(line, hour) if _trips_df is not None else None
    if sched_trip is None:
        sched_trip = _fallback_headway_from_trip_duration(line)
    if sched_trip is None:
        sched_trip = 18.0
    scheduled_eta = int(round(max(2.0, min(90.0, float(sched_trip)))))
    eta = float(scheduled_eta)

    wx_row, _, _ = _resolve_weather_observation(now, line, stop_id)
    if wx_row is not None:
        raw_wx = str(wx_row.get("weather_condition", "clear") or "clear")
        wx_label = _map_weather_display(raw_wx)
        wx_extra, wx_eta_note = _weather_eta_from_observation(wx_row)
        eta += wx_extra
        weather_detail = f"Weather from weather_observations.csv. {wx_eta_note}"
    else:
        wx_label = "Clear"
        weather_detail = (
            "weather_observations.csv: no matching pool; Clear shown; no weather add-on to ETA."
        )
        wx_extra = 0.0

    eta = float(max(2.0, min(90.0, eta)))
    eta_i = int(round(eta))

    flow_sub = _matching_passenger_flow_rows(line, stop_id, now)
    demand_lbl, demand_badge = _crowding_display_and_badge(line, stop_id, now, flow_sub)
    if demand_lbl is None:
        demand_lbl = "Insufficient passenger_flow.csv match for this stop"
        demand_badge = "dem-mod"

    flow_sc, flow_std_agg = _flow_sample_and_waiting_std(flow_sub)
    conf = _confidence_from_real_data(
        pd.DataFrame(),
        now,
        traffic,
        wx_label,
        flow_sample_sum=flow_sc,
        flow_waiting_std_agg=flow_std_agg,
    )

    line_name = _line_name_from_bus_stops(line)
    stop_name = _stop_type_label_for_line_stop(line, stop_id, None)
    shelter_note: str | None = None
    if wx_label == "Rain" and _shelter_available(line, stop_id):
        shelter_note = "Shelter available at this stop."

    rec = _recommendation_from_comparison(scheduled_eta, eta_i, traffic, demand_lbl or "")
    route_path = _line_route_preview(line) or "—"
    lineage = (
        "ETA: bus_trips.csv (median headway as baseline) + weather_observations.csv (delay add-on if matched)\n"
        "Traffic: bus_trips.csv (mode of traffic_level for this line)\n"
        "Weather: weather_observations.csv (or Clear if no pool)\n"
        "Demand: bus_trips.csv (avg_occupancy_pct) + passenger_flow.csv (crowding_level)\n"
        "Confidence: passenger_flow.csv (sample_count, std_passengers_waiting) + traffic/weather penalties\n"
        "Recommendation: derived from ETA vs bus_trips.csv schedule headway"
    )

    return {
        "eta_minutes": eta_i,
        "scheduled_eta_minutes": scheduled_eta,
        "schedule_delta_min": int(eta_i - scheduled_eta),
        "confidence_pct": conf,
        "passenger_demand": demand_lbl,
        "demand_badge": demand_badge,
        "shelter_note": shelter_note,
        "is_transfer_hub": _is_transfer_hub_stop(line, stop_id),
        "recommendation": rec,
        "bus_line": line,
        "stop_id": stop_id,
        "line_name": line_name,
        "stop_name": stop_name,
        "traffic_level": traffic,
        "weather_condition": wx_label,
        "weather_detail": weather_detail,
        "ai_model_used": True,
        "explanation": "No stop_arrivals.csv rows for this line; metrics use other CSVs as listed below.",
        "metric_lineage": lineage,
        "route_path": route_path,
        "peak_traffic_alert": traffic == "Heavy",
        "rush_hour_overlap": traffic == "Heavy" and _is_rush_hour(hour),
        "data_source": "csv_trips_flow_weather",
    }


def _simulate_prediction(line: str, stop_id: str) -> dict[str, Any]:
    """When core CSVs are missing: prefer CSV-only path, else minimal fallback for UI."""
    if _trips_df is not None or _flow_df is not None or _weather_df is not None:
        try:
            return _predict_trips_flow_weather_only(line, stop_id)
        except Exception:
            pass
    now = datetime.now()
    line_name = _line_name_from_bus_stops(line)
    stop_lbl = _stop_type_label_for_line_stop(line, stop_id, None)
    if stop_lbl == stop_id:
        stop_lbl = next((n for sid, n in FALLBACK_STOPS if sid == stop_id), stop_id)
    rp = _line_route_preview(line)
    return {
        "eta_minutes": 12,
        "scheduled_eta_minutes": 12,
        "schedule_delta_min": 0,
        "confidence_pct": 62,
        "passenger_demand": "Required CSV files not loaded",
        "demand_badge": "dem-mod",
        "shelter_note": None,
        "is_transfer_hub": _is_transfer_hub_stop(line, stop_id),
        "recommendation": "Load stop_arrivals.csv, bus_trips.csv, and related files for data-driven results.",
        "bus_line": line,
        "stop_id": stop_id,
        "line_name": line_name,
        "stop_name": stop_lbl,
        "traffic_level": "Moderate",
        "weather_condition": "Clear",
        "weather_detail": "Data files unavailable.",
        "ai_model_used": True,
        "explanation": "Fallback: datasets failed to load.",
        "metric_lineage": "No CSV data loaded; values are placeholders until files are available.",
        "route_path": rp or "—",
        "peak_traffic_alert": False,
        "rush_hour_overlap": False,
        "data_source": "fallback",
    }


def _predict_from_csv(line: str, stop_id: str) -> dict[str, Any]:
    assert _arrivals_df is not None
    df = _arrivals_df
    now = datetime.now()
    hour = now.hour

    use, arrivals_scope = _select_arrivals_use(df, line, stop_id, hour)
    if use.empty:
        return _predict_trips_flow_weather_only(line, stop_id)

    stop_rows = df[(df["line_id"] == line) & (df["stop_id"] == stop_id)]
    label_row = stop_rows.iloc[0] if not stop_rows.empty else None

    mn_series = pd.to_numeric(use["minutes_to_next_bus"], errors="coerce")
    median_next = float(mn_series.median())
    if math.isnan(median_next) and label_row is not None and not stop_rows.empty:
        median_next = float(
            pd.to_numeric(stop_rows["minutes_to_next_bus"], errors="coerce").median()
        )
    if math.isnan(median_next):
        line_mn = pd.to_numeric(df[df["line_id"] == line]["minutes_to_next_bus"], errors="coerce")
        median_next = float(line_mn.median())
    if math.isnan(median_next):
        st = _scheduled_eta_from_trips(line, hour)
        if st is not None:
            median_next = float(st)
        else:
            fb = _fallback_headway_from_trip_duration(line)
            median_next = float(fb) if fb is not None else 18.0

    mean_delay = float(pd.to_numeric(use["delay_min"], errors="coerce").mean())
    if math.isnan(mean_delay):
        mean_delay = 0.0

    traffic = _traffic_level_from_arrivals(use)

    eta = median_next + 0.35 * min(max(mean_delay, 0.0), 20.0)
    eta += _traffic_eta_adjustment(traffic)

    wx_row, _, _ = _resolve_weather_observation(now, line, stop_id)
    if wx_row is not None:
        raw_wx = str(wx_row.get("weather_condition", "clear") or "clear")
        wx_label = _map_weather_display(raw_wx)
        wx_extra, wx_eta_note = _weather_eta_from_observation(wx_row)
        eta += wx_extra
        weather_detail = f"weather_observations.csv — condition {raw_wx}. {wx_eta_note}"
        weather_lineage = "weather_observations.csv (deterministic row for this hour)"
    else:
        wx_ar = _weather_from_arrivals_slice(use)
        if wx_ar is not None:
            wx_label = wx_ar
            weather_detail = (
                f"Weather label from stop_arrivals.csv ({arrivals_scope}) "
                f"via weather_condition (aggregated)."
            )
            weather_lineage = f"stop_arrivals.csv ({arrivals_scope}) — weather_condition"
        else:
            wx_label = "Clear"
            weather_detail = (
                "weather_observations.csv: no pool for this hour; "
                "stop_arrivals slice had no weather_condition. Clear; no extra weather delay."
            )
            weather_lineage = "default Clear (no match in weather_observations.csv or stop_arrivals weather)"

        wx_extra = 0.0

    eta = float(max(2.0, min(55.0, eta)))
    eta_i = int(round(eta))

    sched_trip = _scheduled_eta_from_trips(line, hour) if _trips_df is not None else None
    if sched_trip is not None:
        scheduled_eta = int(round(max(2.0, min(90.0, sched_trip))))
        sched_src = "bus_trips.csv (median gap between planned_departure times)"
    else:
        scheduled_eta = int(round(max(2.0, min(90.0, median_next))))
        sched_src = "stop_arrivals.csv (minutes_to_next_bus median; bus_trips unavailable)"

    flow_sub = _matching_passenger_flow_rows(line, stop_id, now)
    demand_lbl, demand_badge = _crowding_display_and_badge(line, stop_id, now, flow_sub)
    if demand_lbl is None:
        demand_lbl = "No passenger_flow.csv match for this stop and time"
        demand_badge = "dem-mod"

    flow_sc, flow_std_agg = _flow_sample_and_waiting_std(flow_sub)

    conf = _confidence_from_real_data(
        use, now, traffic, wx_label, flow_sample_sum=flow_sc, flow_waiting_std_agg=flow_std_agg
    )

    line_name = _line_name_from_bus_stops(line)
    stop_name = _stop_type_label_for_line_stop(line, stop_id, label_row)

    shelter_note: str | None = None
    if wx_label == "Rain" and _shelter_available(line, stop_id):
        shelter_note = "Shelter available at this stop."

    is_transfer_hub = _is_transfer_hub_stop(line, stop_id)

    rec = _recommendation_from_comparison(scheduled_eta, eta_i, traffic, demand_lbl or "")

    eta_src = (
        f"stop_arrivals.csv — minutes_to_next_bus median, delay_min mean ({arrivals_scope}); "
        f"traffic adjustment from CSV traffic level; "
        f"optional add-on from weather_observations.csv transit_delay_risk / precipitation"
    )
    traffic_src = (
        "stop_arrivals.csv — mode of traffic_level (else delay_min distribution)"
    )
    demand_src = "bus_trips.csv — avg_occupancy_pct; passenger_flow.csv — crowding_level + sample_count"
    conf_src = (
        "stop_arrivals.csv — row count, delay variance, date recency; "
        "passenger_flow.csv — sample_count sum, std_passengers_waiting; "
        "traffic & weather penalties from displayed labels"
    )
    rec_src = "Computed from ETA vs scheduled headway (sources above)"

    lineage = (
        f"ETA: {eta_src}\n"
        f"Traffic: {traffic_src}\n"
        f"Weather: {weather_lineage}\n"
        f"Demand: {demand_src}\n"
        f"Confidence: {conf_src}\n"
        f"Recommendation: {rec_src}\n"
        f"Schedule comparison baseline: {sched_src}"
    )

    route_path = _line_route_preview(line)
    schedule_delta = int(eta_i - scheduled_eta)
    peak_alert = traffic == "Heavy"
    rush_overlap = peak_alert and _is_rush_hour(hour)

    return {
        "eta_minutes": eta_i,
        "scheduled_eta_minutes": scheduled_eta,
        "schedule_delta_min": schedule_delta,
        "confidence_pct": conf,
        "passenger_demand": demand_lbl,
        "demand_badge": demand_badge,
        "shelter_note": shelter_note,
        "is_transfer_hub": is_transfer_hub,
        "recommendation": rec,
        "bus_line": line,
        "stop_id": stop_id,
        "line_name": line_name,
        "stop_name": stop_name,
        "traffic_level": traffic,
        "weather_condition": wx_label,
        "weather_detail": weather_detail,
        "ai_model_used": True,
        "explanation": "All primary metrics are computed from the CSV sources listed under Technical Details.",
        "metric_lineage": lineage,
        "route_path": route_path,
        "peak_traffic_alert": peak_alert,
        "rush_hour_overlap": rush_overlap,
        "data_source": "csv",
    }


def _load_rf_artifact() -> None:
    """Load Random Forest pipeline from model.pkl; leave rule-based path if missing or invalid."""
    global _rf_artifact
    _rf_artifact = None
    path = BASE_DIR / "model.pkl"
    if not path.is_file():
        print(
            "[RF] model.pkl not found — ETA uses rule-based logic only. "
            "Train with: python train_model.py"
        )
        return
    try:
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "pipeline" not in artifact:
            print("[RF] model.pkl has unexpected format — using rule-based ETA only.")
            return
        _rf_artifact = artifact
        m = artifact.get("metrics", {})
        mae, rmse, r2 = m.get("mae"), m.get("rmse"), m.get("r2")
        if mae is not None and rmse is not None and r2 is not None:
            print(
                "[RF] Loaded RandomForestRegressor (model.pkl) | "
                f"validation MAE={float(mae):.4f} min, RMSE={float(rmse):.4f} min, R²={float(r2):.4f}"
            )
        else:
            print("[RF] Loaded RandomForestRegressor from model.pkl.")
    except Exception as e:
        print(f"[RF] Could not load model.pkl ({e!r}) — using rule-based ETA only.")


def _rf_feature_frame(line: str, stop_id: str, now: datetime) -> pd.DataFrame:
    """Single-row feature frame aligned with train_model.py."""
    hour = int(now.hour)
    delay = 0.0
    traffic_raw = "moderate"
    wx_raw = "clear"

    if _arrivals_df is not None and not _arrivals_df.empty:
        sub = _arrivals_df[
            (_arrivals_df["line_id"] == line) & (_arrivals_df["stop_id"] == stop_id)
        ]
        if not sub.empty and "delay_min" in sub.columns:
            dmed = pd.to_numeric(sub["delay_min"], errors="coerce").median()
            delay = float(dmed) if dmed == dmed else 0.0
        if not sub.empty and "traffic_level" in sub.columns:
            tm = sub["traffic_level"].dropna().astype(str).str.lower().str.strip()
            if not tm.empty:
                traffic_raw = str(tm.mode().iloc[0])
        if not sub.empty and "weather_condition" in sub.columns:
            wm = sub["weather_condition"].dropna().astype(str).str.lower().str.strip()
            if not wm.empty:
                wx_raw = str(wm.mode().iloc[0])

    wx_row, _, _ = _resolve_weather_observation(now, line, stop_id)
    if wx_row is not None:
        raw = str(wx_row.get("weather_condition", "") or "").strip().lower()
        if raw:
            wx_raw = raw

    flow_sub = _matching_passenger_flow_rows(line, stop_id, now)
    apw = 0.0
    if not flow_sub.empty and "avg_passengers_waiting" in flow_sub.columns:
        v = pd.to_numeric(flow_sub["avg_passengers_waiting"], errors="coerce").mean()
        if v == v:
            apw = float(v)

    row = {
        "hour_of_day": hour,
        "delay_min": delay,
        "avg_passengers_waiting": apw,
        "line_id": str(line).lower().strip(),
        "stop_id": str(stop_id).strip(),
        "traffic_level": traffic_raw,
        "weather_condition": wx_raw,
    }
    return pd.DataFrame([row])


def _predict_rf_eta_minutes(line: str, stop_id: str, now: datetime) -> float:
    if _rf_artifact is None:
        raise RuntimeError("RF artifact not loaded")
    pipe = _rf_artifact["pipeline"]
    X = _rf_feature_frame(line, stop_id, now)
    y_hat = pipe.predict(X)
    return float(y_hat[0])


def _apply_rf_eta_overrides(pred: dict[str, Any], line: str, stop_id: str) -> dict[str, Any]:
    now = datetime.now()
    raw_eta = _predict_rf_eta_minutes(line, stop_id, now)
    eta_i = int(round(max(2.0, min(90.0, raw_eta))))
    out = dict(pred)
    sched = int(out.get("scheduled_eta_minutes", eta_i))
    out["eta_minutes"] = eta_i
    out["schedule_delta_min"] = int(eta_i - sched)
    out["recommendation"] = _recommendation_from_comparison(
        sched,
        eta_i,
        str(out.get("traffic_level", "Moderate")),
        str(out.get("passenger_demand") or ""),
    )
    mae = 0.0
    if _rf_artifact:
        mae = float(_rf_artifact.get("metrics", {}).get("mae") or 0.0)
    rf_banner = (
        f"ETA: Random Forest Regressor (scikit-learn) trained on stop_arrivals.csv; "
        f"held-out MAE≈{mae:.2f} min.\n"
    )
    out["metric_lineage"] = rf_banner + str(out.get("metric_lineage", ""))
    out["explanation"] = (
        "Primary ETA is predicted by a Random Forest model; "
        "supporting metrics use the CSV/rule logic described below.\n"
        + str(out.get("explanation", ""))
    )
    out["data_source"] = "random_forest"
    return out


def _predict(line: str, stop_id: str) -> dict[str, Any]:
    if _arrivals_df is None:
        pred = _simulate_prediction(line, stop_id)
    else:
        try:
            pred = _predict_from_csv(line, stop_id)
        except Exception:
            pred = _simulate_prediction(line, stop_id)

    if _rf_artifact is not None:
        try:
            pred = _apply_rf_eta_overrides(pred, line, stop_id)
        except Exception as exc:
            print(f"[RF] Model inference failed; using rule-based ETA. ({exc!r})")

    return _enrich_transit_ui(pred, line, stop_id)


_init_data()
_load_rf_artifact()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    sbl = stops_by_line()

    if request.method == "POST":
        line = (request.form.get("bus_line") or "").strip()
        stop_id = (request.form.get("stop_id") or "").strip()
        valid_ids = {sid for sid, _ in sbl.get(line, ())}

        if line not in BUS_LINES:
            error = "Please select a valid bus line."
        elif stop_id not in valid_ids:
            error = "Please select a valid stop for that line."
        else:
            prediction = _predict(line, stop_id)

    html = render_template(
        "index.html",
        line_options=line_options(),
        stops_by_line=sbl,
        route_preview_by_line=route_preview_by_line(),
        prediction=prediction,
        error=error,
        data_note=_load_note,
    )
    resp = make_response(html)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, Pragma: no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.0", port=8000)
