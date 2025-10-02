# pybamm_studio_copilot_min.py
import json, re, io, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import streamlit as st

try:
    import pybamm as pb
except Exception as e:
    st.error(f"PyBaMM import failed: {e}")
    st.stop()

# ---- Minimal local parameter "RAG" store (hard-coded, offline) ----
# Subset resembling Chen2020 LCO/Graphite at ~25 C; values illustrative.
PARAM_DB = [
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

# ---- Tiny schemas ----
@dataclass
class CellSpec:
    chemistry: str = "li-ion"
    positive_active: str = "LCO"
    negative_active: str = "Graphite"
    temperature_C: float = 25.0

@dataclass
class Step:
    type: str
    rate: Optional[str] = None
    until_voltage_V: Optional[float] = None
    until_current_C: Optional[float] = None
    rest_min: Optional[float] = None

@dataclass
class ExperimentSpec:
    steps: List[Step]
    temperature_C: float = 25.0
    repeats: int = 1

# ---- NL parsers (rule-based, minimal) ----
def parse_cell(text: str) -> CellSpec:
    chem = "li-ion"
    pos = "LCO" if re.search(r"\bLCO\b|cobalt", text, re.I) else "LCO"
    neg = "Graphite" if re.search(r"\bgraphite\b", text, re.I) else "Graphite"
    t = re.search(r"(\d+(\.\d+)?)\s*°?\s*C", text, re.I)
    temp = float(t.group(1)) if t else 25.0
    return CellSpec(chemistry=chem, positive_active=pos, negative_active=neg, temperature_C=temp)

def parse_experiment(text: str) -> ExperimentSpec:
    # Supports: "charge at 0.5C to 4.2V", "cv hold at 4.2V until 0.05C", "rest 30 min", "discharge at 1C to 3.0V", "repeat 3"
    steps: List[Step] = []
    for m in re.finditer(r"(charge|discharge)\s+at\s+(-?\d+(\.\d+)?)\s*C\s*(to|until)\s*(\d+(\.\d+)?)\s*V", text, re.I):
        kind = m.group(1).lower()
        rate = f"{m.group(2)}C"
        v = float(m.group(5))
        steps.append(Step(type="CC_CHARGE" if kind=="charge" else "CC_DISCHARGE", rate=rate, until_voltage_V=v))
    for m in re.finditer(r"\brest\s+(\d+(\.\d+)?)\s*(min|mins|minute|minutes|h|hr|hours)\b", text, re.I):
        val = float(m.group(1))
        unit = m.group(3).lower()
        rest_min = val*60 if unit.startswith('h') else val
        steps.append(Step(type="REST", rest_min=rest_min))
    for m in re.finditer(r"\bcv\s*(hold)?\s*(at|to)\s*(\d+(\.\d+)?)\s*V\s*until\s*(\d+(\.\d+)?)\s*C", text, re.I):
        v = float(m.group(3)); c = float(m.group(5))
        steps.append(Step(type="CV", until_voltage_V=v, until_current_C=c))
    rep = 1
    mrep = re.search(r"\brepeat\s+(\d+)\b", text, re.I)
    if mrep: rep = int(mrep.group(1))
    # default if nothing parsed
    if not steps:
        steps = [Step(type="CC_CHARGE", rate="1C", until_voltage_V=4.2),
                 Step(type="REST", rest_min=10),
                 Step(type="CC_DISCHARGE", rate="1C", until_voltage_V=3.0)]
    return ExperimentSpec(steps=steps, repeats=rep)

# ---- Local "RAG": filter PARAM_DB by CellSpec ----
def retrieve_params(cell: CellSpec) -> Dict[str, Any]:
    matches = [p for p in PARAM_DB if p["chem"]==cell.chemistry and p["pos"]==cell.positive_active and p["neg"]==cell.negative_active]
    # Map to PyBaMM ParameterValues keys
    param_values = {p["symbol"]: p["value"] for p in matches}
    citations = [{"name": p["name"], "symbol": p["symbol"], "units": p["units"], "source": p["source"]} for p in matches]
    return {"values": param_values, "citations": citations}

# ---- Build pybamm.Experiment from ExperimentSpec ----
def to_pybamm_experiment(es: ExperimentSpec) -> pb.Experiment:
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
    # repeat
    if es.repeats > 1:
        lines.append(f"Repeat for {es.repeats} times")
    return pb.Experiment(lines, temperature=es.temperature_C+273.15)

# ---- Run model ----
def run_model(param_pack: Dict[str, Any], experiment: pb.Experiment):
    model = pb.lithium_ion.SPM()  # minimal, fast
    params = pb.ParameterValues("Chen2020")  # baseline
    # overlay local retrieved values if present
    for k, v in param_pack.get("values", {}).items():
        try:
            params.update({k: v})
        except Exception:
            pass
    sim = pb.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    return sim, sol

# ---- Streamlit UI ----
st.set_page_config(page_title="PyBaMM Studio • Copilot (Mini)", layout="wide")
st.title("Copilot (Minimal Vertical Slice)")

with st.sidebar:
    st.subheader("Describe your cell and experiment")
    default_text = ("LCO graphite coin cell at 25 C. "
                    "Charge at 0.5C to 4.2V, cv hold at 4.2V until 0.05C, rest 30 min, "
                    "discharge at 1C to 3.0V, repeat 3")
    user_text = st.text_area("Natural language input", value=default_text, height=180)
    run_btn = st.button("Parse • Retrieve • Build • Run", type="primary")

col1, col2 = st.columns([1,1])

if run_btn:
    cell = parse_cell(user_text)
    exp = parse_experiment(user_text)
    pack = retrieve_params(cell)
    with col1:
        st.markdown("**Parsed CellSpec**")
        st.json(asdict(cell))
        st.markdown("**Parsed ExperimentSpec**")
        st.json({"steps":[asdict(s) for s in exp.steps], "repeats":exp.repeats, "temperature_C":exp.temperature_C})
        st.markdown("**Retrieved Parameters (local)**")
        st.json(pack["citations"])
        try:
            pb_exp = to_pybamm_experiment(exp)
            st.markdown("**pybamm.Experiment**")
            st.code("\n".join(pb_exp.operating_conditions()), language="text")
        except Exception as e:
            st.error(f"Experiment build error: {e}")
    with col2:
        try:
            pb_exp = to_pybamm_experiment(exp)
            sim, sol = run_model(pack, pb_exp)
            fig1 = sim.plot(["Voltage [V]"], testing=True)
            st.pyplot(fig1)
            # Repro-Pack export
            repro = {
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
                "pybamm_version": pb.__version__,
                "model": "SPM",
                "cell_spec": asdict(cell),
                "experiment_spec": {"steps":[asdict(s) for s in exp.steps], "repeats":exp.repeats, "temperature_C":exp.temperature_C},
                "parameters": pack["citations"],
                "experiment_lines": pb_exp.operating_conditions(),
            }
            buf = io.BytesIO(json.dumps(repro, indent=2).encode())
            st.download_button("Download Repro-Pack (.json)", data=buf, file_name="repro_pack.json", mime="application/json")
        except Exception as e:
            st.error(f"Run failed: {e}")
