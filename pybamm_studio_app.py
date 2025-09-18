import streamlit as st
import pybamm
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="PyBaMM Studio", layout="wide")

st.title("🔋 PyBaMM Studio - Vertical Slice")

# Sidebar configuration
st.sidebar.header("Model Configuration")

# Select model
model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ["SPM", "SPMe", "DFN"],
    index=0
)

# Select parameter set
param_choice = st.sidebar.selectbox(
    "Choose Parameter Set:",
    ["Chen2020", "Marquis2019", "Mohtat2020"]
)

# Simulation time
sim_time = st.sidebar.slider("Simulation Time (hours)", 0.1, 5.0, 1.0, 0.1)

# Step 1: Load model
if model_choice == "SPM":
    model = pybamm.lithium_ion.SPM()
elif model_choice == "SPMe":
    model = pybamm.lithium_ion.SPMe()
else:
    model = pybamm.lithium_ion.DFN()

# Step 2: Load parameter values
params = pybamm.ParameterValues(param_choice)

# Step 3: Create experiment (simple GCD cycle)
experiment = pybamm.Experiment([
    ("Discharge at 1C until 2.5V",
     "Charge at 1C until 4.2V",
     "Rest for 30 minutes")
] * 1)  # one cycle

# Utility function: Export Repro-Pack
def export_repro_pack(sim, filename="repro_pack.json"):
    """Export simulation configuration and results as a JSON Repro-Pack."""
    sim.export_json(filename)
    return filename

# Step 4: Run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation... This may take a moment."):
        sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
        sol = sim.solve()

        # Plot voltage vs time
        fig, ax = plt.subplots()
        sol.plot(ax=ax, variables=["Voltage [V]"])
        st.pyplot(fig)

        # Display summary
        st.success("Simulation complete!")
        st.write(f"Final time: {sol['Time [s]'].entries[-1]/3600:.2f} h")
        st.write(f"Final voltage: {sol['Voltage [V]'].entries[-1]:.2f} V")

        # Export Repro-Pack and Plot
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export JSON
            repro_file = os.path.join(tmpdir, "repro_pack.json")
            export_repro_pack(sim, repro_file)
            with open(repro_file, "rb") as f:
                st.download_button(
                    label="Download Repro-Pack",
                    data=f,
                    file_name="repro_pack.json",
                    mime="application/json"
                )

            # Export PNG plot
            plot_file = os.path.join(tmpdir, "voltage_vs_time.png")
            fig.savefig(plot_file, dpi=300)
            with open(plot_file, "rb") as f:
                st.download_button(
                    label="Download Voltage vs Time Plot",
                    data=f,
                    file_name="voltage_vs_time.png",
                    mime="image/png"
                )
else:
    st.info("Configure your model in the sidebar and click 'Run Simulation' to begin.")
