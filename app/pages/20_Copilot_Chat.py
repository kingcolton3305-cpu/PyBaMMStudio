import os, json, io, re, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import streamlit as st
from groq import Groq

import pybamm as pb

# --- tiny local "RAG" store (kept in-repo, free) ---
PARAM_DB = [
    {"chem":"li-ion","pos":"LCO","neg":"Graphite",
     "symbol":"Nominal cell capacity [A.h]","value":2.0,"units":"A.h","src":"Chen2020 subset"},
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"R_p","value":5.0e-6,"units":"m","src":"Chen2020 subset"},
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"R_n","value":6.0e-6,"units":"m","src":"Chen2020 subset"},
    {"chem":"li-ion","pos":"LCO","neg":"Graphite","symbol":"T_ref","value":298.15,"units":"K","src":"Chen2020 subset"},
]

# --- schemas ---
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
    rate: Optional[str] = None
    until_voltage_V: Optional[float] = None
    until_current_C: Optional[float] = None
    rest_min: Optional[float] = None

@dataclass
class ExperimentSpec:
    steps: List[Step]
    repeats: int = 1
    temperature_C: float = 25.0

# --- helpers ---
SYS = """You are Mixtral AI inside PyBaMM Studio. Output ONLY JSON when asked to build.
When user describes an experiment, return:
{
 "action":"build",
 "cell":{"chemistry": "...","positive_active":"...","negative_active":"...","temperature_C": 25},
 "experiment":{"temperature_C":25,"repeats":1,"steps":[
   {"kind":"CC","rate":"0.5C","until_voltage_V":4.2},
   {"kind":"CV","until_voltage_V":4.2,"until_current_C":0.05},
   {"kind":"Rest","rest_min":30},
   {"kind":"CC","rate":"-1C","until_voltage_V":3.0}
 ]}
}
If not building, reply with {"action":"chat","text":"..."}.
Do not include code fences.
"""

def get_client():
    key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Missing GROQ_API_KEY (set in .streamlit/secrets.toml).")
        st.stop()
    return Groq(api_key=key)

def call_mixtral(messages):
    client = get_client()
    resp = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def retrieve_params(cell: CellSpec) -> Dict[str, Any]:
    hits = [p for p in PARAM_DB
            if p["chem"]==cell.chemistry and p["pos"]==cell.positive_active and p["neg"]==cell.negative_active]
    return {
        "values": {p["symbol"]: p["value"] for p in hits},
        "citations": [{"symbol":p["symbol"],"units":p["units"],"src":p["src"]} for p in hits]
    }

def to_pybamm_experiment(es: ExperimentSpec) -> pb.Experiment:
    lines = []
    for s in es.steps:
        if s.kind=="CC":
            # allow negative rate for discharge
            lines.append(f"{'Discharge' if str(s.rate).startswith('-') else 'Charge'} at {s.rate} until {float(s.until_voltage_V)} V")
        elif s.kind=="CV":
            lines.append(f"Hold at {float(s.until_voltage_V)} V until {float(s.until_current_C)} C")
        elif s.kind=="Rest":
            lines.append(f"Rest for {int(s.rest_min)} minutes")
    if es.repeats>1:
        lines.append(f"Repeat for {es.repeats} times")
    # temp on Experiment ctor expects Kelvin
    return pb.Experiment(lines, temperature=es.temperature_C+273.15), lines

def run_model(param_pack: Dict[str,Any], experiment: pb.Experiment):
    model = pb.lithium_ion.SPM()  # fast and cloud-friendly
    params = pb.ParameterValues("Chen2020")
    # Overlay
    for k,v in param_pack.get("values",{}).items():
        try:
            params.update({k:v})
        except Exception:
            pass
    sim = pb.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    return sim, sol

# --- UI ---
st.set_page_config(page_title="Copilot Chat • Mixtral", layout="wide")
st.title("Copilot Chat (Mixtral)")

with st.sidebar:
    st.caption("Free, cloud-friendly. Mixtral via Groq API. Local parameter RAG.")
    mode = st.radio("Mode", ["Chat", "Build & Run"], horizontal=True)
    temperature = st.slider("Model temperature", 0.0, 1.0, 0.2, 0.1, disabled=True)

if "history" not in st.session_state:
    st.session_state.history = [{"role":"system","content":SYS}]

# chat UI
for m in st.session_state.get("display", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Describe your cell and experiment… e.g. LCO/graphite 25°C, CC-CV to 4.2V, rest, 3 cycles.")
if prompt:
    st.session_state.history.append({"role":"user","content":prompt})
    if mode=="Chat":
        out = call_mixtral(st.session_state.history)
        st.session_state.display = st.session_state.get("display", []) + [
            {"role":"user","content":prompt},
            {"role":"assistant","content":out},
        ]
        st.rerun()
    else:
        # Ask the model to return build JSON
        build_query = st.session_state.history + [{"role":"user","content":"Return JSON for build now."}]
        raw = call_mixtral(build_query)
        # try parse
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            st.error("Model did not return JSON. Switch to Chat to debug the prompt.")
        if data and data.get("action")=="build":
            cell = CellSpec(**data["cell"])
            expspec = ExperimentSpec(
                temperature_C=data["experiment"].get("temperature_C", cell.temperature_C),
                repeats=data["experiment"].get("repeats",1),
                steps=[Step(**s) for s in data["experiment"]["steps"]]
            )
            pack = retrieve_params(cell)
            pb_exp, lines = to_pybamm_experiment(expspec)

            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader("Parsed Specs")
                st.json({"cell":asdict(cell),"experiment":asdict(expspec)})
                st.subheader("Local Parameters")
                st.json(pack["citations"])
                st.subheader("pybamm.Experiment lines")
                st.code("\n".join(lines))

            with col2:
                try:
                    sim, sol = run_model(pack, pb_exp)
                    fig = sim.plot(["Voltage [V]"], testing=True)
                    st.pyplot(fig)
                    repro = {
                        "timestamp": dt.datetime.utcnow().isoformat()+"Z",
                        "pybamm_version": pb.__version__,
                        "model":"SPM",
                        "experiment_lines": lines,
                        "cell_spec": asdict(cell),
                        "parameters": pack["citations"],
                    }
                    buf = io.BytesIO(json.dumps(repro, indent=2).encode())
                    st.download_button("Download Repro-Pack (.json)", data=buf, file_name="repro_pack.json", mime="application/json")
                except Exception as e:
                    st.error(f"Run failed: {e}")
        else:
            st.error("No build action detected. Try refining your description.")
