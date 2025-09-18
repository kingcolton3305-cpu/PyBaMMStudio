import streamlit as st
import pybamm
import matplotlib.pyplot as plt

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
else:
    st.info("Configure your model in the sidebar and click 'Run Simulation' to begin.")
