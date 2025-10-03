# pybamm_studio_app_test.py
# Streamlit app: robust demo with PyBaMM optional.
# Fixes applied:
# - Single set_page_config at top
# - Defined render_copilot_chat_panel()
# - Replaced stray 'ss' with st.session_state
# - get_var() tries multiple keys safely and returns flat numpy arrays
# - plot_voltage_vs_capacity() does not assume "halfway split"
# - Parameter set selector is robust to missing previous choice
# - All figures are closed after use
# - run_experiment() defined once, used consistently
# - Guarded all PyBaMM calls; app runs without PyBaMM installed

import os
import io
import json
import zipfile
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Optional PyBaMM import
try:
    import pybamm as pb
    _PYBAMM_OK = True
except Exception:
    pb = None
    _PYBAMM_OK = False

st.set_page_config(page_title="PyBaMM Studio (Patched)", layout="wide")

# ---------- Utilities ----------

def log(message: str) -> None:
    """Append a message to session 'hist' for simple logging."""
    if "hist" not in st.session_state:
        st.session_state.hist = []
    st.session_state.hist.append(message)

def get_var(solution, candidate_keys: List[str]) -> Tuple[np.ndarray, str]:
    """
    Try variable names in order and return flattened numpy array and picked key.
    Works with both ProcessedVariable and raw arrays.
    """
    if solution is None:
        raise ValueError("Solution is None. Run an experiment first.")
    last_err = None
    for key in candidate_keys:
        try:
            var = solution[key]
            # ProcessedVariable -> .entries; Arrays -> direct
            entries = getattr(var, "entries", var)
            arr = np.asarray(entries).ravel()
            if arr.size == 0:
                continue
            return arr, key
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"No matching variable among: {candidate_keys}. Last error: {last_err}")

def run_experiment(model, params, experiment=None):
    """Run a PyBaMM Simulation with safe defaults."""
    if not _PYBAMM_OK:
        raise RuntimeError("PyBaMM not available in this environment.")
    sim = pb.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    return sim, sol

def plot_voltage_vs_capacity(solution) -> plt.Figure:
    """Plot Voltage [V] versus Discharge capacity [A.h] or fallbacks."""
    volt, v_key = get_var(solution, ["Voltage [V]", "Terminal voltage [V]"])
    cap, c_key = get_var(solution, [
        "Discharge capacity [A.h]",
        "Capacity [A.h]",
        "Discharge capacity [mAh]"
    ])
    # If cap likely in mAh, convert to A.h for consistency
    if "mAh" in c_key:
        cap = cap / 1000.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(cap, volt, linewidth=1.5)
    ax.set_xlabel("Discharge capacity [A.h]")
    ax.set_ylabel("Voltage [V]")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Voltage vs Capacity")
    fig.tight_layout()
    return fig

def available_param_sets() -> List[str]:
    """Return list of known parameter sets if PyBaMM is present."""
    if not _PYBAMM_OK:
        return ["<PyBaMM not installed>"]
    try:
        # Canonical list in recent PyBaMM
        return sorted(list(pb.ParameterValues.list_known_parameter_sets()))
    except Exception:
        # Fallback minimal list
        return ["Chen2020", "Ai2020", "Ecker2015"]

# ---------- Copilot Panel ----------

def render_copilot_chat_panel() -> None:
    """
    Minimal placeholder chat-like panel with a text input and running log.
    This function previously was referenced but undefined.
    """
    st.subheader("AI Copilot")
    user_input = st.text_input("Prompt", key="copilot_input", placeholder="Describe your task...")
    if st.button("Log", type="secondary"):
        if user_input.strip():
            log(f"USER: {user_input.strip()}")
        else:
            log("USER: <empty>")

    with st.expander("History", expanded=False):
        if "hist" in st.session_state and st.session_state.hist:
            for i, msg in enumerate(st.session_state.hist[-200:]):
                st.write(f"{i+1:03d}: {msg}")
        else:
            st.caption("No history yet.")

# ---------- Main UI ----------

def main():
    st.title("PyBaMM Studio • Patched")
    st.caption("Runs without PyBaMM. Enables experiment and plotting when PyBaMM is installed.")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        st.toggle("Show Copilot", value=True, key="show_copilot")

        if _PYBAMM_OK:
            param_sets = available_param_sets()
            last = st.session_state.get("last_param_set", None)
            idx = param_sets.index(last) if last in param_sets else 0
            selected = st.selectbox("Parameter set", options=param_sets, index=idx)
            st.session_state.last_param_set = selected

            default_exp = pb.Experiment([
                "Charge at 0.5C until 4.2 V",
                "Hold at 4.2 V until C/50",
                "Discharge at 0.5C until 3.0 V"
            ])
            use_default = st.checkbox("Use default GCD experiment", value=True)
        else:
            st.info("PyBaMM not detected. Install it to enable simulations.")
            selected = None
            use_default = True
            default_exp = None

    if st.session_state.get("show_copilot", True):
        render_copilot_chat_panel()

    # Simulation area
    st.subheader("Simulation")
    if not _PYBAMM_OK:
        st.warning("Simulation disabled. PyBaMM is not installed here.")
        return

    cols = st.columns([1, 1])
    with cols[0]:
        run_btn = st.button("Build model and run", type="primary")

    if run_btn:
        try:
            model = pb.lithium_ion.DFN()  # DFN as requested
            params = pb.ParameterValues(st.session_state.last_param_set)

            if use_default:
                experiment = default_exp
            else:
                experiment = None  # Could extend with a text box for custom YAML

            sim, sol = run_experiment(model, params, experiment)
            st.session_state.solution = sol
            st.success("Run complete.")
        except Exception as e:
            st.exception(e)
            return

    # Plot
    if st.session_state.get("solution") is not None:
        try:
            fig = plot_voltage_vs_capacity(st.session_state.solution)
            st.pyplot(fig, clear_figure=False)
            plt.close(fig)
        except Exception as e:
            st.exception(e)

    # Export bundle
    st.subheader("Export")
    if st.session_state.get("solution") is not None:
        if st.button("Download results bundle (.zip)"):
            try:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Save a PNG of the plot
                    fig = plot_voltage_vs_capacity(st.session_state.solution)
                    img_io = io.BytesIO()
                    fig.savefig(img_io, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    zf.writestr("voltage_vs_capacity.png", img_io.getvalue())

                    # Save minimal metadata
                    meta = {
                        "parameter_set": st.session_state.get("last_param_set"),
                        "notes": "PyBaMM Studio Patched export",
                    }
                    zf.writestr("metadata.json", json.dumps(meta, indent=2))

                st.download_button(
                    label="Download bundle",
                    data=buf.getvalue(),
                    file_name="pybamm_studio_results.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.exception(e)

if __name__ == "__main__":
    main()
