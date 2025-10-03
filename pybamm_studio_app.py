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

def plot_current_vs_time(solution):
    """
    Plot the current vs. time for a PyBaMM solution.

    Args:
        solution (pybamm.Solution): The solution object from a PyBaMM simulation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    time = solution.t
    current = solution["Current [A]"].data

    ax.plot(time, current, label="Current", color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Current vs. Time")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

def plot_voltage_vs_capacity(solution):
    """
    Plot the voltage vs. capacity for a PyBaMM solution.

    Args:
        solution (pybamm.Solution): The solution object from a PyBaMM simulation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    capacity = solution["Capacity [A.h]"].data
    voltage = solution["Terminal voltage [V]"].data

    ax.plot(capacity, voltage, label="Voltage", color="red")
    ax.set_xlabel("Capacity (A.h)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs. Capacity")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


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

            st.subheader("Current vs. Time")
            plot_current_vs_time(solution)

            st.subheader("Voltage vs. Capacity")
            plot_voltage_vs_capacity(solution)

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

# ==== Copilot Chat (Mixtral) • Panel ====
import os, json, io, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import streamlit as st
import requests
import pybamm as pb

# ---------- Tiny local param "RAG" subset (free, offline) ----------
_PARAM_DB = [
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"Nominal cell capacity [A.h]","value":2.0,"units":"A.h","src":"Chen2020 subset"},
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"R_p","value":5.0e-6,"units":"m","src":"Chen2020 subset"},
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"R_n","value":6.0e-6,"units":"m","src":"Chen2020 subset"},
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"T_ref","value":298.15,"units":"K","src":"Chen2020 subset"},
]

# ---------- Schemas ----------
@dataclass
class CellSpec:
    chemistry: str = "li-ion"
    positive_active: str = "LCO"
    negative_active: str = "Graphite"
    temperature_C: float = 25.0
    notes: Optional[str] = None

@dataclass
class Step:
    kind: str  # "CC","CV","Rest"
    rate: Optional[str] = None         # e.g. "0.5C" or "-1C"
    until_voltage_V: Optional[float] = None
    until_current_C: Optional[float] = None
    rest_min: Optional[float] = None

@dataclass
class ExperimentSpec:
    steps: List[Step]
    repeats: int = 1
    temperature_C: float = 25.0

# ---------- Mistral client (HTTP, no SDK) ----------
_SYS = """You are Mixtral inside PyBaMM Studio. When asked to build, return ONLY JSON of this form:
{"action":"build","cell":{"chemistry":"li-ion","positive_active":"LCO","negative_active":"Graphite","temperature_C":25},
 "experiment":{"temperature_C":25,"repeats":1,"steps":[
   {"kind":"CC","rate":"0.5C","until_voltage_V":4.2},
   {"kind":"CV","until_voltage_V":4.2,"until_current_C":0.05},
   {"kind":"Rest","rest_min":30},
   {"kind":"CC","rate":"-1C","until_voltage_V":3.0}
 ]}}
If not building, reply with {"action":"chat","text":"..."} only.
"""

def _mistral_key() -> str:
    return st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY") or ""

def _mixtral(messages: List[Dict[str,str]]) -> str:
    key = _mistral_key()
    if not key:
        return "Missing MISTRAL_API_KEY"
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {"model":"open-mixtral-8x7b","messages":messages,"temperature":0.2}
    r = requests.post(url,
                      headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"},
                      data=json.dumps(payload),
                      timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ---------- Helpers ----------
def _params_for(cell: CellSpec) -> Dict[str,Any]:
    hits = [p for p in _PARAM_DB if p["chem"]==cell.chemistry and p["pos"]==cell.positive_active and p["neg"]==cell.negative_active]
    return {
        "values": {p["symbol"]: p["value"] for p in hits},
        "citations": [{"symbol":p["symbol"],"units":p["units"],"src":p["src"]} for p in hits]
    }

def _to_experiment(ex: ExperimentSpec) -> tuple[pb.Experiment, List[str]]:
    lines: List[str] = []
    for s in ex.steps:
        if s.kind == "CC":
            lines.append(f"{'Discharge' if str(s.rate).startswith('-') else 'Charge'} at {s.rate} until {float(s.until_voltage_V)} V")
        elif s.kind == "CV":
            lines.append(f"Hold at {float(s.until_voltage_V)} V until {float(s.until_current_C)} C")
        elif s.kind == "Rest":
            lines.append(f"Rest for {int(s.rest_min)} minutes")
    if ex.repeats > 1:
        lines.append(f"Repeat for {ex.repeats} times")
    exp = pb.Experiment(lines, temperature=ex.temperature_C + 273.15)
    return exp, lines

def _run(pack: Dict[str,Any], experiment: pb.Experiment):
    model = pb.lithium_ion.SPM()
    params = pb.ParameterValues("Chen2020")
    for k, v in pack.get("values", {}).items():
        try:
            params.update({k: v})
        except Exception:
            pass
    sim = pb.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    return sim, sol

def _emit_pybamm_script(model_name: str, lines: List[str], overlay: Dict[str,Any], temp_C: float) -> str:
    overlay_json = json.dumps(overlay or {}, indent=2)
    lines_json = json.dumps(lines)
    script = f'''# Auto-generated by Copilot Chat
import pybamm as pb

_MODEL_NAME = "{model_name}"
_OVERLAY = {overlay_json}

def build_model():
    return getattr(pb.lithium_ion, _MODEL_NAME)()

def get_parameter_values():
    params = pb.ParameterValues("Chen2020")
    try:
        params.update(_OVERLAY)
    except Exception:
        pass
    if "Nominal cell capacity [A.h]" not in params:
        try:
            cap = _OVERLAY.get("Nominal cell capacity [A.h]", 2.0)
            params.update({{"Nominal cell capacity [A.h]": cap}})
        except Exception:
            pass
    return params

def solve(model, params, experiment=None):
    if experiment is None:
        lines = {lines_json}
        experiment = pb.Experiment(lines, temperature={temp_C}+273.15)
    sim = pb.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    return sim, sol
'''
    return script

# ---------- UI ----------
st.set_page_config(page_title="PyBaMM Copilot Chat", layout="wide")
st.title("PyBaMM Copilot")

# Sidebar collapsible chat
with st.sidebar.expander("Copilot Chat (Mixtral)", expanded=False):
    st.caption("Provide a description. Click Build & Run to execute. Set MISTRAL_API_KEY in Secrets.")
    default_prompt = ("LCO/graphite at 25°C. CC 0.5C→4.2V, CV to 0.05C, Rest 30 min, "
                      "Discharge −1C→3.0V, 3 cycles.")
    user = st.text_area("Prompt", default_prompt, height=120)
    chat_btn = st.button("Chat")
    build_btn = st.button("Build & Run", type="primary")
    gen_btn = st.button("Generate Studio Script")

# Session chat history
if "hist" not in st.session_state:
    st.session_state.hist = [{"role":"system","content":_SYS}]

# Output placeholders
left, right = st.columns([1,1])

# Chat
if chat_btn:
    st.session_state.hist.append({"role":"user","content":user})
    try:
        out = _mixtral(st.session_state.hist)
    except Exception as e:
        out = f"Chat error: {e}"
    with right:
        st.subheader("Chat")
        st.write(out)

# Build & Run
if build_btn:
    query = st.session_state.hist + [{"role":"user","content":user},
                                     {"role":"user","content":"Return JSON for build now."}]
    try:
        raw = _mixtral(query)
        data = json.loads(raw)
    except Exception as e:
        with right:
            st.error(f"Model JSON error: {e}")
        data = None

    if data and data.get("action") == "build":
        try:
            cell = CellSpec(**data["cell"])
            ex = ExperimentSpec(
                temperature_C=data["experiment"].get("temperature_C", cell.temperature_C),
                repeats=data["experiment"].get("repeats", 1),
                steps=[Step(**s) for s in data["experiment"]["steps"]],
            )
            pack = _params_for(cell)
            pb_exp, lines = _to_experiment(ex)

            with left:
                st.subheader("Specs")
                st.json({"cell": asdict(cell), "experiment": asdict(ex)})
                st.subheader("Local parameters")
                st.json(pack["citations"])
                st.subheader("Experiment lines")
                st.code("\n".join(lines))

            with right:
                sim, sol = _run(pack, pb_exp)
                fig = sim.plot(["Voltage [V]"], testing=True)
                st.pyplot(fig)
                repro = {
                    "timestamp": dt.datetime.utcnow().isoformat()+"Z",
                    "pybamm_version": pb.__version__,
                    "model": "SPM",
                    "experiment_lines": lines,
                    "cell_spec": asdict(cell),
                    "parameters": pack["citations"],
                }
                buf = io.BytesIO(json.dumps(repro, indent=2).encode())
                st.download_button("Download Repro-Pack (.json)", data=buf,
                                   file_name="repro_pack.json", mime="application/json")
        except Exception as e:
            with right:
                st.error(f"Run failed: {e}")
    else:
        with right:
            st.error("No build action in response. Use Chat to refine prompt.")

# Generate Studio-ready script (does not run; safe without model call)
if gen_btn:
    # Try to reuse last built specs if available; else parse via model once.
    if "last_lines" in st.session_state and "last_overlay" in st.session_state and "last_tempC" in st.session_state:
        lines = st.session_state["last_lines"]
        overlay = st.session_state["last_overlay"]
        tempC = st.session_state["last_tempC"]
    else:
        # One-shot build request for script emission
        query = st.session_state.hist + [{"role":"user","content":user},
                                         {"role":"user","content":"Return JSON for build now."}]
        try:
            raw = _mixtral(query)
            data = json.loads(raw)
            cell = CellSpec(**data["cell"])
            ex = ExperimentSpec(
                temperature_C=data["experiment"].get("temperature_C", cell.temperature_C),
                repeats=data["experiment"].get("repeats", 1),
                steps=[Step(**s) for s in data["experiment"]["steps"]],
            )
            pack = _params_for(cell)
            _, lines = _to_experiment(ex)
            overlay, tempC = pack.get("values", {}), ex.temperature_C
        except Exception as e:
            with right:
                st.error(f"Cannot generate script: {e}")
            lines, overlay, tempC = [], {}, 25.0

    code_str = _emit_pybamm_script(model_name="SPM", lines=lines, overlay=overlay, temp_C=tempC)
    with left:
        st.subheader("PyBaMM Studio script")
        st.code(code_str, language="python")
        st.download_button("Download script (pybamm_script.py)",
                           data=code_str.encode(), file_name="pybamm_script.py",
                           mime="text/x-python")

# ==== End Copilot Chat panel ====

choice = st.sidebar.radio("View", ["Studio","Copilot Chat"], horizontal=True)
if choice=="Copilot Chat":
    render_copilot_chat_panel()

st.caption("© PyBaMM Studio Vertical Slice – Streamlit")
