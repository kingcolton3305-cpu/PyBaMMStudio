# app.py
# PyBaMM Studio – Streamlit Vertical Slice
# Panels: Code | Parameters | Run & Visualize | Export
# Notes:
# - Works even if PyBaMM isn't installed (UI loads; Run disabled until available)
# - Parameter panel can load any built-in PyBaMM parameter set and lets you edit values
# - Exports a text-based "Repro-Pack" through a download button (no file picker)

from __future__ import annotations
import io, json, sys, platform, hashlib, zipfile
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Try importing PyBaMM -----------------------------------------------------
try:
    import pybamm as pb
    _PYBAMM_OK = True
    _PYBAMM_VERSION = pb.__version__
except Exception as e:
    pb = None  # type: ignore
    _PYBAMM_OK = False
    _PYBAMM_VERSION = None
    _PYBAMM_ERR = f"{type(e).__name__}: {e}"

st.set_page_config(page_title="PyBaMM Studio (Streamlit)", layout="wide")

# --- Starter Script -----------------------------------------------------------
STARTER_SCRIPT = """import pybamm as pb

def build_model():
    # Single Particle Model (try SPMe() or DFN() if desired)
    return pb.lithium_ion.SPM()

def get_parameter_values():
    # Prefer name-based lookup; fallback to legacy attribute in older versions
    try:
        return pb.ParameterValues("Chen2020")
    except Exception:
        return pb.ParameterValues(pb.parameter_sets.Chen2020)

def run_experiment():
    return pb.Experiment([
        ("Charge at 1C until 4.2V", "Hold at 4.2V until C/50"),
        ("Rest for 5 minutes",),
        ("Discharge at 1C until 3.0V"),
    ])

def solve(model, parameter_values, experiment=None):
    sim = pb.Simulation(model, parameter_values=parameter_values, experiment=experiment)
    sol = sim.solve()
    return sim, sol
"""

# --- Helpers ------------------------------------------------------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def parse_jsonlike(s: str) -> Any:
    """Try json.loads; if it fails, return the raw string."""
    try:
        return json.loads(s)
    except Exception:
        return s

def dict_to_df(d: Dict[str, Any]) -> pd.DataFrame:
    # Show values as JSON strings for safe editing
    rows = [{"key": k, "value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v} for k, v in d.items()]
    return pd.DataFrame(rows, columns=["key", "value"]).sort_values("key").reset_index(drop=True)

def df_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        k = str(row["key"]).strip()
        if not k:
            continue
        out[k] = parse_jsonlike(str(row["value"]))
    return out

def load_param_set(name: str) -> Dict[str, Any]:
    if not _PYBAMM_OK:
        raise RuntimeError("PyBaMM not available.")
    params = pb.ParameterValues(name)
    # Convert to plain Python types/strings for editing
    # Many values print nicely; we stringify to be safe with editor
    return {k: str(v) for k, v in params.items()}

def run_user_script(code: str, overrides: Dict[str, Any]) -> Tuple[Any, Any]:
    """Execute user code (expects build_model/get_parameter_values/run_experiment/solve),
    apply overrides, return (sim, sol)."""
    if not _PYBAMM_OK:
        raise RuntimeError("PyBaMM not available.")

    ns: Dict[str, Any] = {"pb": pb}
    exec(compile(code, "<user_script>", "exec"), ns, ns)

    build_model = ns.get("build_model")
    get_parameter_values = ns.get("get_parameter_values")
    run_experiment = ns.get("run_experiment", lambda: None)
    solve = ns.get("solve")

    if not callable(build_model) or not callable(get_parameter_values) or not callable(solve):
        raise RuntimeError("Script must define build_model(), get_parameter_values(), and solve(model, params, experiment=None)")

    model = build_model()
    params = get_parameter_values()

    # Wrap dict into ParameterValues if needed
    if isinstance(params, dict):
        params = pb.ParameterValues(params)

    # Apply overrides
    for k, v in overrides.items():
        try:
            params[k] = v
        except Exception as e:
            st.warning(f"Could not override '{k}': {e}")

    experiment = run_experiment()
    sim, sol = solve(model, params, experiment)
    return sim, sol

def voltage_from_solution(sol) -> Tuple[np.ndarray, np.ndarray]:
    try:
        var = sol["Voltage [V]"]
    except KeyError:
        var = sol["Terminal voltage [V]"]
    t = sol.t
    v = np.array(var.entries).reshape(-1)
    return t, v

def render_repro_text(script: str, params: Dict[str, Any], sim, sol) -> str:
    lines = []
    lines.append("# PyBaMM Studio Repro-Pack (text)")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append(f"Python: {sys.version.split()[0]} ({sys.platform})")
    lines.append(f"PyBaMM: {_PYBAMM_VERSION if _PYBAMM_OK else 'unavailable'}")
    lines.append(f"Script SHA1: {sha1(script)}")
    lines.append("")
    lines.append("## Parameters (JSON)")
    lines.append(json.dumps(params, indent=2))
    lines.append("")
    lines.append("## Code")
    lines.append(script)
    # Optional: add a short summary if a solution exists
    if sol is not None:
        try:
            t, v = voltage_from_solution(sol)
            lines.append("")
            lines.append("## Summary")
            lines.append(f"n_time={t.size}, V_min={float(v.min()):.3f}, V_max={float(v.max()):.3f}")
        except Exception:
            pass
    return "\n".join(lines)

# --- Session state ------------------------------------------------------------
if "code" not in st.session_state:
    st.session_state.code = STARTER_SCRIPT
if "param_df" not in st.session_state:
    st.session_state.param_df = pd.DataFrame(columns=["key", "value"])
if "solution" not in st.session_state:
    st.session_state.solution = None
if "simulation" not in st.session_state:
    st.session_state.simulation = None
if "last_param_set" not in st.session_state:
    st.session_state.last_param_set = "Chen2020"

# --- Sidebar: Environment -----------------------------------------------------
st.sidebar.title("Environment")
if _PYBAMM_OK:
    st.sidebar.success(f"PyBaMM: v{_PYBAMM_VERSION}")
else:
    st.sidebar.error("PyBaMM unavailable")
    st.sidebar.caption(f"{_PYBAMM_ERR}")

st.sidebar.write("Python:", sys.version.split()[0])
st.sidebar.write("Platform:", platform.platform())

st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** If PyBaMM isn’t available on Streamlit Cloud, pin Python 3.10–3.12 and install:\n\n"
                    "`pip install \"pybamm[jax,plot]\"`")

# --- Layout: Tabs -------------------------------------------------------------
tabs = st.tabs(["1) Code", "2) Parameters", "3) Run & Visualize", "4) Export"])

# --- Tab 1: Code --------------------------------------------------------------
with tabs[0]:
    st.subheader("Editable PyBaMM Script")
    st.caption("Your script must define: build_model(), get_parameter_values(), optional run_experiment(), and solve(model, params, experiment=None).")
    st.session_state.code = st.text_area(
        "Python code", value=st.session_state.code, height=400, label_visibility="collapsed", key="code_editor"
    )
    st.code(st.session_state.code, language="python")

# --- Tab 2: Parameters --------------------------------------------------------
with tabs[1]:
    st.subheader("Parameter Inspector")
    cols_top = st.columns([1,1,2,2])
    with cols_top[0]:
        # Pre-populate with a few common sets plus any discovered in pb.parameter_sets
        default_sets = ["Chen2020", "Marquis2019", "Ai2020", "OKane2022"]
        all_sets = default_sets.copy()
        if _PYBAMM_OK:
            try:
                for s in sorted(getattr(pb.parameter_sets, "__all__", [])):
                    if s not in all_sets:
                        all_sets.append(s)
            except Exception:
                pass
        selected = st.selectbox("Parameter set", options=all_sets, index=all_sets.index(st.session_state.last_param_set) if st.session_state.last_param_set in all_sets else 0)
    with cols_top[1]:
        load_clicked = st.button("Load set into table", disabled=not _PYBAMM_OK)
    with cols_top[2]:
        search = st.text_input("Search keys (live filter)", value="")
    with cols_top[3]:
        st.write("")

    if load_clicked and _PYBAMM_OK:
        try:
            raw = load_param_set(selected)
            st.session_state.last_param_set = selected
            st.session_state.param_df = dict_to_df(raw)
            st.success(f"Loaded {selected} with {len(st.session_state.param_df)} entries.")
        except Exception as e:
            st.error(f"Could not load set '{selected}': {e}")

    # Live filter
    df_view = st.session_state.param_df
    if search.strip():
        mask = df_view["key"].str.contains(search, case=False, na=False)
        df_view = df_view.loc[mask].reset_index(drop=True)

    st.caption("Double-click cells to edit. Values accept JSON (numbers, strings, booleans).")
    edited_df = st.data_editor(df_view, use_container_width=True, num_rows="dynamic", key="param_editor")
    # Write edits back to full table if no search, otherwise just update the visible rows
    if search.strip():
        # Merge edited rows back into the full table
        full = st.session_state.param_df.set_index("key")
        part = edited_df.set_index("key")
        full.update(part)
        st.session_state.param_df = full.reset_index()
    else:
        st.session_state.param_df = edited_df

    st.download_button(
        "Download parameters.json",
        data=json.dumps(df_to_dict(st.session_state.param_df), indent=2).encode("utf-8"),
        file_name="parameters.json",
        mime="application/json",
    )

# --- Tab 3: Run & Visualize ---------------------------------------------------
with tabs[2]:
    st.subheader("Run Model and Visualize")
    run_btn = st.button("Run", type="primary", disabled=not _PYBAMM_OK)
    if not _PYBAMM_OK:
        st.info("Install PyBaMM with JAX to enable running: `pip install \"pybamm[jax,plot]\"`")

    log = st.empty()
    plot_area = st.empty()

    if run_btn and _PYBAMM_OK:
        try:
            overrides = df_to_dict(st.session_state.param_df)
            sim, sol = run_user_script(st.session_state.code, overrides)
            st.session_state.simulation = sim
            st.session_state.solution = sol

            # Plot Voltage vs Time
            t, v = voltage_from_solution(sol)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, v)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Voltage [V]")
            ax.set_title("Voltage vs Time")
            ax.grid(True, alpha=0.3)
            plot_area.pyplot(fig, clear_figure=True)

            log.success(f"[ok] Solve complete. n_time={t.size}, V in [{float(v.min()):.3f}, {float(v.max()):.3f}] V")
        except Exception as e:
            log.error(f"[error] {type(e).__name__}: {e}")

    elif st.session_state.solution is not None and _PYBAMM_OK:
        # Show last plot if available
        try:
            t, v = voltage_from_solution(st.session_state.solution)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, v)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Voltage [V]")
            ax.set_title("Voltage vs Time")
            ax.grid(True, alpha=0.3)
            plot_area.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Plot unavailable: {e}")

# --- Tab 4: Export ------------------------------------------------------------
with tabs[3]:
    st.subheader("Export Repro-Pack (Text or ZIP)")

    # Always produce a text repro-pack (no picker; downloadable)
    text_blob = render_repro_text(
        script=st.session_state.code,
        params=df_to_dict(st.session_state.param_df),
        sim=st.session_state.simulation,
        sol=st.session_state.solution,
    )
    st.download_button(
        "Download repro_pack.txt",
        data=text_blob.encode("utf-8"),
        file_name=f"repro_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # Optional: also provide a ZIP containing the text and a PNG if solution exists
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("repro_pack.txt", text_blob)

        # Save a plot if solution exists
        if _PYBAMM_OK and st.session_state.solution is not None:
            try:
                t, v = voltage_from_solution(st.session_state.solution)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(t, v)
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Voltage [V]")
                ax.set_title("Voltage vs Time")
                ax.grid(True, alpha=0.3)
                png_io = io.BytesIO()
                fig.savefig(png_io, format="png", bbox_inches="tight")
                plt.close(fig)
                zf.writestr("plots/voltage_vs_time.png", png_io.getvalue())
            except Exception as e:
                zf.writestr("plots/README.txt", f"Plot could not be generated: {e}")

        # Basic metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pybamm": _PYBAMM_VERSION if _PYBAMM_OK else None,
            "script_sha1": sha1(st.session_state.code),
        }
        zf.writestr("run_metadata.json", json.dumps(meta, indent=2))

    st.download_button(
        "Download repro_pack.zip",
        data=zip_bytes.getvalue(),
        file_name=f"repro_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

st.caption("© PyBaMM Studio Vertical Slice – Streamlit")


# ==== Copilot (Minimal) • injected block ====
# Names are prefixed to avoid collisions with existing app symbols.
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import io as _cop_io
import json as _cop_json
import datetime as _cop_dt
import re as _cop_re

import streamlit as st as _cop_st

try:
    import pybamm as _cop_pb
except Exception as _e:
    # If PyBaMM isn't available, surface a small warning but do not crash the host app.
    try:
        _cop_st.warning(f"Copilot disabled: PyBaMM import failed: {_e}")
    except Exception:
        pass
else:
    # Tiny local param DB (no network). Values illustrative.
    _COP_PARAM_DB = [
        {"chem":"li-ion", "pos":"LCO", "neg":"Graphite",
         "name":"Nominal cell capacity [A.h]", "symbol":"Nominal cell capacity [A.h]",
         "value":2.0, "units":"A.h", "source":"Chen2020 subset"},
        {"chem":"li-ion", "pos":"LCO", "neg":"Graphite",
         "name":"Positive particle radius [m]", "symbol":"R_p",
         "value":5.0e-6, "units":"m", "source":"Chen2020 subset"},
        {"chem":"li-ion", "pos":"LCO", "neg":"Graphite",
         "name":"Negative particle radius [m]", "symbol":"R_n",
         "value":6.0e-6, "units":"m", "source":"Chen2020 subset"},
        {"chem":"li-ion", "pos":"LCO", "neg":"Graphite",
         "name":"Ambient temperature [K]", "symbol":"T_ref",
         "value":298.15, "units":"K", "source":"Chen2020 subset"},
    ]

    @dataclass
    class _CopCell:
        chemistry: str = "li-ion"
        positive_active: str = "LCO"
        negative_active: str = "Graphite"
        temperature_C: float = 25.0

    @dataclass
    class _CopStep:
        type: str
        rate: Optional[str] = None
        until_voltage_V: Optional[float] = None
        until_current_C: Optional[float] = None
        rest_min: Optional[float] = None

    @dataclass
    class _CopExp:
        steps: List[_CopStep]
        temperature_C: float = 25.0
        repeats: int = 1

    def _cop_parse_cell(text: str) -> _CopCell:
        pos = "LCO" if _cop_re.search(r"\\bLCO\\b|cobalt", text, _cop_re.I) else "LCO"
        neg = "Graphite" if _cop_re.search(r"\\bgraphite\\b", text, _cop_re.I) else "Graphite"
        t = _cop_re.search(r"(\\d+(\\.\\d+)?)\\s*°?\\s*C", text, _cop_re.I)
        temp = float(t.group(1)) if t else 25.0
        return _CopCell(chemistry="li-ion", positive_active=pos, negative_active=neg, temperature_C=temp)

    def _cop_parse_exp(text: str) -> _CopExp:
        steps: List[_CopStep] = []
        for m in _cop_re.finditer(r"(charge|discharge)\\s+at\\s+(-?\\d+(\\.\\d+)?)\\s*C\\s*(to|until)\\s*(\\d+(\\.\\d+)?)\\s*V", text, _cop_re.I):
            kind = m.group(1).lower()
            rate = f"{m.group(2)}C"
            v = float(m.group(5))
            steps.append(_CopStep(type="CC_CHARGE" if kind=="charge" else "CC_DISCHARGE", rate=rate, until_voltage_V=v))
        for m in _cop_re.finditer(r"\\brest\\s+(\\d+(\\.\\d+)?)\\s*(min|mins|minute|minutes|h|hr|hours)\\b", text, _cop_re.I):
            val = float(m.group(1)); unit = m.group(3).lower()
            rest_min = val*60 if unit.startswith('h') else val
            steps.append(_CopStep(type="REST", rest_min=rest_min))
        for m in _cop_re.finditer(r"\\bcv\\s*(hold)?\\s*(at|to)\\s*(\\d+(\\.\\d+)?)\\s*V\\s*until\\s*(\\d+(\\.\\d+)?)\\s*C", text, _cop_re.I):
            v = float(m.group(3)); c = float(m.group(5))
            steps.append(_CopStep(type="CV", until_voltage_V=v, until_current_C=c))
        rep = 1
        mrep = _cop_re.search(r"\\brepeat\\s+(\\d+)\\b", text, _cop_re.I)
        if mrep: rep = int(mrep.group(1))
        if not steps:
            steps = [_CopStep(type="CC_CHARGE", rate="1C", until_voltage_V=4.2),
                     _CopStep(type="REST", rest_min=10),
                     _CopStep(type="CC_DISCHARGE", rate="1C", until_voltage_V=3.0)]
        return _CopExp(steps=steps, repeats=rep)

    def _cop_retrieve_params(cell: _CopCell) -> Dict[str, Any]:
        matches = [p for p in _COP_PARAM_DB if p["chem"]==cell.chemistry and p["pos"]==cell.positive_active and p["neg"]==cell.negative_active]
        return {
            "values": {p["symbol"]: p["value"] for p in matches},
            "citations": [{"name": p["name"], "symbol": p["symbol"], "units": p["units"], "source": p["source"]} for p in matches],
        }

    def _cop_to_pybamm_exp(es: _CopExp) -> _cop_pb.Experiment:
        lines: List[str] = []
        for s in es.steps:
            if s.type == "CC_CHARGE":
                lines.append(f"Charge at {s.rate} until {s.until_voltage_V} V")
            elif s.type == "CC_DISCHARGE":
                lines.append(f"Discharge at {s.rate} until {s.until_voltage_V} V")
            elif s.type == "REST":
                lines.append(f"Rest for {int(s.rest_min)} minutes")
            elif s.type == "CV":
                lines.append(f"Hold at {s.until_voltage_V} V until {s.until_current_C} C")
        if es.repeats > 1:
            lines.append(f"Repeat for {es.repeats} times")
        return _cop_pb.Experiment(lines, temperature=es.temperature_C+273.15)

    def _cop_run(param_pack: Dict[str, Any], experiment: _cop_pb.Experiment):
        model = _cop_pb.lithium_ion.SPM()
        params = _cop_pb.ParameterValues("Chen2020")
        for k, v in param_pack.get("values", {}).items():
            try:
                params.update({k: v})
            except Exception:
                pass
        sim = _cop_pb.Simulation(model, parameter_values=params, experiment=experiment)
        sol = sim.solve()
        return sim, sol

    # Sidebar expander UI. Output shows in main area placeholders.
    _copilot_main = _cop_st.container()
    with _cop_st.sidebar.expander("Copilot (Minimal)", expanded=False):
        _default_text = ("LCO graphite coin cell at 25 C. "
                         "Charge at 0.5C to 4.2V, cv hold at 4.2V until 0.05C, rest 30 min, "
                         "discharge at 1C to 3.0V, repeat 3")
        _txt = _cop_st.text_area("Natural language experiment", value=_default_text, height=160, key="copilot_text_min")
        _run = _cop_st.button("Run Copilot", key="copilot_run_min")
    if _run:
        try:
            _cell = _cop_parse_cell(_txt)
            _exp = _cop_parse_exp(_txt)
            _pack = _cop_retrieve_params(_cell)
            _py_exp = _cop_to_pybamm_exp(_exp)
            with _copilot_main:
                _cop_st.subheader("Copilot • Result")
                c1, c2 = _cop_st.columns([1,1])
                with c1:
                    _cop_st.markdown("**CellSpec**")
                    _cop_st.json(asdict(_cell))
                    _cop_st.markdown("**ExperimentSpec**")
                    _cop_st.json({"steps":[asdict(s) for s in _exp.steps], "repeats":_exp.repeats, "temperature_C":_exp.temperature_C})
                    _cop_st.markdown("**Parameters**")
                    _cop_st.json(_pack["citations"])
                    _cop_st.markdown("**pybamm.Experiment**")
                    _cop_st.code("\\n".join(_py_exp.operating_conditions()), language="text")
                with c2:
                    _sim, _sol = _cop_run(_pack, _py_exp)
                    _fig = _sim.plot(["Voltage [V]"], testing=True)
                    _cop_st.pyplot(_fig)
                    _repro = {
                        "timestamp": _cop_dt.datetime.utcnow().isoformat() + "Z",
                        "pybamm_version": _cop_pb.__version__,
                        "model": "SPM",
                        "cell_spec": asdict(_cell),
                        "experiment_spec": {"steps":[asdict(s) for s in _exp.steps], "repeats":_exp.repeats, "temperature_C":_exp.temperature_C},
                        "parameters": _pack["citations"],
                        "experiment_lines": _py_exp.operating_conditions(),
                    }
                    _buf = _cop_io.BytesIO(_cop_json.dumps(_repro, indent=2).encode())
                    _cop_st.download_button("Download Repro-Pack (.json)", data=_buf, file_name="repro_pack.json", mime="application/json", key="copilot_reprop")
        except Exception as _err:
            try:
                _cop_st.error(f"Copilot failed: {_err}")
            except Exception:
                pass
# ==== End Copilot (Minimal) injection ====
