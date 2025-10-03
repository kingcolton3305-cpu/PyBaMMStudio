# pybamm_studio_app.py
# Streamlit vertical slice for "PyBaMM Studio"
# Requirements: streamlit, pybamm, numpy, requests
from __future__ import annotations

import io
import json
import re
import sys
import math
import traceback
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import streamlit as st

try:
    import pybamm as pb
except Exception as e:
    st.error("PyBaMM import failed. Install pybamm in this environment.")
    raise

# ---------- Page setup ----------
st.set_page_config(page_title="PyBaMM Studio — Vertical Slice", layout="wide")
st.title("PyBaMM Studio — Vertical Slice")
st.caption(
    f"Python {sys.version.split()[0]} • PyBaMM {getattr(pb,'__version__','unknown')}"
)

# ---------- Dataclass and helpers ----------
@dataclass
class Step:
    kind: str  # 'charge'|'discharge'|'rest'|'hold'
    rate: Optional[float] = None             # C-rate (for charge/discharge)
    until_voltage_V: Optional[float] = None  # V threshold
    until_current_C: Optional[float] = None  # expressed as C-fraction (e.g., 1/50 -> 0.02)
    rest_min: Optional[float] = None         # minutes

NUMERIC_RX = re.compile(r"^\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*$")

def _coerce(s: str):
    """Best-effort type coercion: float, int, bool, then stripped string."""
    if s is None:
        return s
    s2 = s.strip()
    if not s2:
        return ""
    # bool
    low = s2.lower()
    if low in {"true", "false"}:
        return low == "true"
    # numeric
    if NUMERIC_RX.match(s2):
        try:
            val = float(s2)
            # keep int if integral
            if math.isfinite(val) and abs(val - round(val)) < 1e-12:
                return int(round(val))
            return val
        except Exception:
            pass
    return s2

def parse_kv_block(text: str) -> Dict[str, object]:
    """
    Parse key:value overrides safely.
    Ignores blank lines and comments (# ...).
    Uses .partition(':') to avoid unpacking errors.
    """
    result: Dict[str, object] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if sep != ":":
            # Not a key-value line; ignore safely
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        result[key] = _coerce(value)
    return result

def _parse_c_fraction(token: str) -> Optional[float]:
    """
    Accept tokens like 'C/50', 'C/20', case-insensitive.
    Returns numeric fraction (1/50 -> 0.02).
    """
    t = token.strip().upper().replace(" ", "")
    if t.startswith("C/"):
        try:
            denom = float(t[2:])
            if denom > 0:
                return 1.0 / denom
        except Exception:
            return None
    return None

# Regex patterns for steps
RX_CHG = re.compile(
    r"^\s*Charge\s+at\s+([0-9.+-eE]+)\s*C\s+until\s+([0-9.+-eE]+)\s*V\s*$",
    re.IGNORECASE,
)
RX_DCH = re.compile(
    r"^\s*Discharge\s+at\s+([0-9.+-eE]+)\s*C\s+until\s+([0-9.+-eE]+)\s*V\s*$",
    re.IGNORECASE,
)
RX_REST = re.compile(
    r"^\s*Rest\s+for\s+([0-9.+-eE]+)\s*(min|mins|minute|minutes)\s*$",
    re.IGNORECASE,
)
RX_HOLD = re.compile(
    r"^\s*Hold\s+at\s+([0-9.+-eE]+)\s*V\s+until\s+(C\s*/\s*[0-9.+-eE]+)\s*$",
    re.IGNORECASE,
)

def parse_steps(text: str) -> List[Step]:
    """
    Parse human-readable experiment steps into Step dataclasses.
    Unknown lines are ignored without crashing.
    """
    steps: List[Step] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        m = RX_CHG.match(line)
        if m:
            rate = float(m.group(1))
            v = float(m.group(2))
            steps.append(Step(kind="charge", rate=rate, until_voltage_V=v))
            continue

        m = RX_DCH.match(line)
        if m:
            rate = float(m.group(1))
            v = float(m.group(2))
            steps.append(Step(kind="discharge", rate=rate, until_voltage_V=v))
            continue

        m = RX_REST.match(line)
        if m:
            mins = float(m.group(1))
            steps.append(Step(kind="rest", rest_min=mins))
            continue

        m = RX_HOLD.match(line)
        if m:
            v = float(m.group(1))
            cfrac = _parse_c_fraction(m.group(2))
            if cfrac:
                steps.append(Step(kind="hold", until_voltage_V=v, until_current_C=cfrac))
            continue

        # silently ignore unknown lines
    return steps

def _steps_to_lines(steps: List[Step]) -> List[str]:
    """
    Convert parsed Steps to PyBaMM Experiment lines.
    """
    lines: List[str] = []
    for s in steps:
        if s.kind == "charge" and s.rate is not None and s.until_voltage_V is not None:
            lines.append(f"Charge at {s.rate}C until {s.until_voltage_V}V")
        elif s.kind == "discharge" and s.rate is not None and s.until_voltage_V is not None:
            lines.append(f"Discharge at {s.rate}C until {s.until_voltage_V}V")
        elif s.kind == "rest" and s.rest_min is not None:
            mins = s.rest_min
            unit = "minutes" if abs(mins - 1.0) > 1e-12 else "minute"
            lines.append(f"Rest for {mins} {unit}")
        elif s.kind == "hold" and s.until_voltage_V is not None and s.until_current_C is not None:
            # invert fraction back to "C/xx"
            frac = s.until_current_C
            if frac > 0:
                denom = round(1.0 / frac)
                lines.append(f"Hold at {s.until_voltage_V}V until C/{int(denom)}")
    return lines

def emit_script(lines: List[str], overrides: Dict[str, object], tempC: float) -> str:
    """
    Emit a full Python script string with the required functions to recreate the experiment.
    """
    safe_overrides = json.dumps(overrides, indent=2)
    exp_lines_json = json.dumps(lines, indent=2)
    tempK = float(tempC) + 273.15

    script = f'''# Auto-generated by PyBaMM Studio — Vertical Slice
import pybamm as pb

def build_model():
    """Return the DFN model."""
    return pb.lithium_ion.DFN()

def get_parameter_values():
    """Return Chen2020 parameters with user overrides applied."""
    params = pb.ParameterValues("Chen2020")
    # User overrides (exact keys must match PyBaMM parameter names)
    overrides = {safe_overrides}
    for k, v in overrides.items():
        try:
            params.update({{k: v}})
        except Exception:
            # ignore invalid keys silently to keep script robust
            pass
    return params

def solve(model, params, experiment=None):
    \"\"\"Solve the simulation with an optional Experiment.\"\"\"
    sim = pb.Simulation(model, parameter_values=params, experiment=experiment)
    return sim.solve()

def build_experiment():
    lines = {exp_lines_json}
    # temperature is Kelvin
    return pb.Experiment(lines, temperature={tempK})

# Example usage when run as a script
if __name__ == "__main__":
    model = build_model()
    params = get_parameter_values()
    exp = build_experiment()
    sol = solve(model, params, exp)
    # Print a tiny summary
    print("Solve complete. t_final = ", float(sol.t[-1]))
'''
    return script

def _find_first_key(container, candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in container:
            return key
    # Fallback: substring search
    for key in container.keys():
        for c in candidates:
            if c.lower() in key.lower():
                return key
    return None

def _extract_var(sol, key: str) -> Optional[np.ndarray]:
    try:
        v = sol[key]
        # ProcessedVariable has .entries
        if hasattr(v, "entries"):
            return np.array(v.entries).reshape(-1)
        # sometime returns scalar
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.array(v).reshape(-1)
        # fallback numeric
        try:
            return np.array([float(v)])
        except Exception:
            return None
    except Exception:
        return None

def _compute_capacity_Ah(t_s: np.ndarray, i_A: np.ndarray) -> np.ndarray:
    """
    Numerical integral of |I| over time to approximate capacity in A·h.
    """
    if t_s.size < 2 or i_A.size < 2:
        return np.zeros_like(t_s, dtype=float)
    # cumtrapz-like manual to avoid SciPy dependency
    dt = np.diff(t_s)
    i_mid = 0.5 * (np.abs(i_A[:-1]) + np.abs(i_A[1:]))
    ah = np.concatenate([[0.0], np.cumsum(i_mid * dt) / 3600.0])
    return ah

# ---------- Copilot (Mixtral) ----------
def call_mixtral(prompt: str, system: str = "You are a helpful PyBaMM code generator.") -> str:
    """
    Call a Mixtral-compatible API using st.secrets.
    Expected secrets format:
    st.secrets["mixtral"]["api_key"]
    st.secrets["mixtral"]["endpoint"] (default https://api.mistral.ai/v1/chat/completions)
    st.secrets["mixtral"]["model"] (default open-mixtral-8x7b)
    """
    import requests

    api_key = None
    endpoint = "https://api.mistral.ai/v1/chat/completions"
    model = "open-mixtral-8x7b"

    try:
        if "mixtral" in st.secrets:
            api_key = st.secrets["mixtral"].get("api_key")
            endpoint = st.secrets["mixtral"].get("endpoint", endpoint)
            model = st.secrets["mixtral"].get("model", model)
    except Exception:
        pass

    if not api_key:
        return "Mixtral API key not found in secrets. Add st.secrets['mixtral']['api_key']."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }

    try:
        r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        # Mistral API: choices[0].message.content
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content or "No content returned by Mixtral."
    except requests.HTTPError as e:
        return f"Mixtral API HTTP error: {e}"
    except Exception as e:
        return f"Mixtral API call failed: {e}"

# ---------- UI Layout ----------
DEFAULT_STEPS = """Charge at 0.5C until 4.2V
Rest for 5 min
Discharge at 1C until 3.0V
"""

with st.sidebar:
    st.header("Controls")
    tempC = st.number_input("Temperature (°C)", value=25.0, step=1.0, format="%.1f")
    st.write("Overrides (key:value per line)")
    overrides_text = st.text_area(
        "Parameter overrides",
        value="Nominal cell capacity [A.h]: 5.0\nSEI resistivity [Ohm.m2]: 2.5e-6",
        height=140,
        label_visibility="collapsed",
    )
    st.markdown(
        "Unknown keys are ignored. Uses safe `partition(':')`. Comments start with `#`."
    )

left, right = st.columns([0.46, 0.54])

# ----- Left: Copilot -----
with left:
    st.subheader("AI Copilot (Mixtral)")
    st.write(
        "Describe your experiment or ask for code. "
        "Use **Send** to chat, or **Build & Run** to execute the generated script."
    )
    copilot_prompt = st.text_area(
        "Message to Copilot",
        value=(
            "Create a PyBaMM DFN script using Chen2020. "
            "Experiment:\n"
            "Charge at 0.5C until 4.2V\n"
            "Rest for 5 min\n"
            "Discharge at 1C until 3.0V\n"
            "Temperature 25C.\n"
            "Define build_model(), get_parameter_values(), and solve(...)."
        ),
        height=220,
    )
    c1, c2 = st.columns(2)
    copilot_output_placeholder = st.empty()
    with c1:
        if st.button("Send"):
            resp = call_mixtral(copilot_prompt)
            copilot_output_placeholder.markdown("**Copilot Response:**\n\n```\n" + resp + "\n```")
    with c2:
        if st.button("Build & Run"):
            resp = call_mixtral(copilot_prompt)
            copilot_output_placeholder.markdown("**Copilot Response:**\n\n```\n" + resp + "\n```")
            # Try to execute returned code safely
            try:
                # Heuristic: extract code block if present
                code_match = re.search(r"```(?:python)?\s*(.*?)```", resp, re.DOTALL | re.IGNORECASE)
                code_text = code_match.group(1) if code_match else resp

                # Minimal sandbox namespace
                ns: Dict[str, object] = {"__name__": "__copilot__", "pb": pb}
                exec(code_text, ns, ns)

                # Expect functions
                if not all(k in ns for k in ("build_model", "get_parameter_values", "solve")):
                    st.error("Generated code missing required functions. Expected build_model, get_parameter_values, solve.")
                else:
                    model = ns["build_model"]()
                    params = ns["get_parameter_values"]()
                    # Try to find an experiment builder or compose a simple one
                    exp = None
                    if "build_experiment" in ns:
                        exp = ns["build_experiment"]()
                    else:
                        exp = pb.Experiment(
                            ["Charge at 0.5C until 4.2V", "Rest for 5 minutes", "Discharge at 1C until 3.0V"],
                            temperature=float(tempC) + 273.15,
                        )
                    sol = ns["solve"](model, params, exp)

                    st.success("Copilot script ran successfully.")
                    # Quick plot: voltage vs time
                    try:
                        # Extract variables safely
                        time_key = _find_first_key(sol, ["Time [s]", "Time [h]"])
                        volt_key = _find_first_key(sol, ["Voltage [V]", "Terminal voltage [V]"])
                        curr_key = _find_first_key(sol, ["Current [A]"])
                        if time_key and volt_key:
                            t = _extract_var(sol, time_key)
                            v = _extract_var(sol, volt_key)
                            if t is not None and v is not None:
                                st.line_chart({"time": t, "voltage [V]": v}, x="time")
                        if time_key and curr_key:
                            t = _extract_var(sol, time_key)
                            i = _extract_var(sol, curr_key)
                            if t is not None and i is not None:
                                st.line_chart({"time": t, "current [A]": i}, x="time")
                    except Exception:
                        st.info("Plots unavailable for generated script.")
            except Exception:
                st.error("Copilot-generated code failed to run. Check the response content.")

# ----- Right: Tabs -----
with right:
    tabs = st.tabs(["Code", "Parameters", "Run & Visualize", "Export"])

    # --- Code tab: studio script generation
    with tabs[0]:
        st.subheader("Generate Studio Script")
        steps_text = st.text_area(
            "Experiment steps",
            value=DEFAULT_STEPS,
            height=140,
        )
        if st.button("Generate Studio Script"):
            try:
                steps = parse_steps(steps_text)
                lines = _steps_to_lines(steps)
                overrides = parse_kv_block(overrides_text)
                script_text = emit_script(lines, overrides, tempC)
                st.code(script_text, language="python")

                st.download_button(
                    "Download script (pybamm_script.py)",
                    data=script_text,
                    file_name="pybamm_script.py",
                    mime="text/x-python",
                )
            except Exception:
                st.error("Script generation failed. Verify your inputs.")

    # --- Parameters tab
    with tabs[1]:
        st.subheader("Resolved Parameters Preview")
        try:
            params = pb.ParameterValues("Chen2020")
            ov = parse_kv_block(overrides_text)
            # Only preview valid updates
            valid_updates = {}
            for k, v in ov.items():
                try:
                    # test applying on a copy to avoid mutating base preview
                    tmp = params.copy()
                    tmp.update({k: v})
                    valid_updates[k] = v
                except Exception:
                    pass

            preview = {"Override key": list(valid_updates.keys()), "Value": list(valid_updates.values())}
            if preview["Override key"]:
                st.write("Overrides that match known parameter keys:")
                st.dataframe(preview, use_container_width=True)
            else:
                st.info("No valid overrides recognized against Chen2020.")
        except Exception:
            st.error("Failed to load Chen2020 parameters.")

    # --- Run & Visualize tab
    with tabs[2]:
        st.subheader("Run Simulation")
        steps_text_run = st.text_area(
            "Experiment steps (for run)",
            value=DEFAULT_STEPS,
            height=140,
        )
        if st.button("Build & Run (Studio)"):
            try:
                # Parse steps and overrides
                steps = parse_steps(steps_text_run)
                lines = _steps_to_lines(steps)
                if not lines:
                    st.error("No valid steps parsed. Use the supported step formats.")
                else:
                    st.write("Experiment lines:")
                    st.code("\n".join(lines), language="text")

                    exp = pb.Experiment(lines, temperature=float(tempC) + 273.15)

                    model = pb.lithium_ion.DFN()
                    params = pb.ParameterValues("Chen2020")
                    # Apply overrides safely
                    ov = parse_kv_block(overrides_text)
                    for k, v in ov.items():
                        try:
                            params.update({k: v})
                        except Exception:
                            # silently ignore bad keys
                            pass

                    sim = pb.Simulation(model, parameter_values=params, experiment=exp)
                    sol = sim.solve()
                    st.success("Simulation completed.")

                    # Extract variables
                    time_key = _find_first_key(sol, ["Time [s]", "Time [h]"])
                    volt_key = _find_first_key(sol, ["Voltage [V]", "Terminal voltage [V]"])
                    curr_key = _find_first_key(sol, ["Current [A]"])
                    if not time_key or not volt_key:
                        st.error("Could not locate time or voltage variables in solution.")
                    else:
                        t = _extract_var(sol, time_key)
                        v = _extract_var(sol, volt_key)
                        if t is None or v is None:
                            st.error("Variable extraction failed.")
                        else:
                            # Current
                            i = _extract_var(sol, curr_key) if curr_key else None
                            # Capacity from I and t
                            cap = _compute_capacity_Ah(t, i) if i is not None else np.linspace(0, 0, t.size)

                            st.markdown("**Voltage vs Time**")
                            st.line_chart({"time": t, "Voltage [V]": v}, x="time")

                            if i is not None:
                                st.markdown("**Current vs Time**")
                                st.line_chart({"time": t, "Current [A]": i}, x="time")
                            else:
                                st.info("Current not available in solution.")

                            st.markdown("**Voltage vs Capacity**")
                            st.line_chart({"Capacity [A·h]": cap, "Voltage [V]": v}, x="Capacity [A·h]")

                    # Stash objects for export
                    st.session_state["last_lines"] = lines
                    st.session_state["last_overrides"] = ov
                    st.session_state["last_tempC"] = float(tempC)
                    st.session_state["last_solution_ok"] = True

            except Exception as e:
                st.session_state["last_solution_ok"] = False
                st.error("Run failed. Check inputs and overrides.")

    # --- Export tab
    with tabs[3]:
        st.subheader("Export")
        # Repro-Pack JSON
        if st.button("Create Repro-Pack JSON"):
            try:
                lines = st.session_state.get("last_lines", _steps_to_lines(parse_steps(DEFAULT_STEPS)))
                ov = st.session_state.get("last_overrides", parse_kv_block(overrides_text))
                tempC_val = float(st.session_state.get("last_tempC", tempC))
                repro = {
                    "timestamp_utc": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
                    "pybamm_version": getattr(pb, "__version__", "unknown"),
                    "model": "DFN",
                    "parameter_set": "Chen2020",
                    "experiment_lines": lines,
                    "overrides": ov,
                    "temperature_C": tempC_val,
                }
                data = json.dumps(repro, indent=2)
                st.code(data, language="json")
                st.download_button(
                    "Download Repro-Pack (.json)",
                    data=data.encode("utf-8"),
                    file_name="repro_pack.json",
                    mime="application/json",
                )
            except Exception:
                st.error("Failed to generate Repro-Pack.")

        # Script export repeating generation with current state
        if st.button("Generate Studio Script (from last run or inputs)"):
            try:
                lines = st.session_state.get("last_lines", _steps_to_lines(parse_steps(DEFAULT_STEPS)))
                ov = st.session_state.get("last_overrides", parse_kv_block(overrides_text))
                tempC_val = float(st.session_state.get("last_tempC", tempC))
                script_text = emit_script(lines, ov, tempC_val)
                st.code(script_text, language="python")
                st.download_button(
                    "Download script (pybamm_script.py)",
                    data=script_text,
                    file_name="pybamm_script.py",
                    mime="text/x-python",
                )
            except Exception:
                st.error("Failed to generate script export.")
