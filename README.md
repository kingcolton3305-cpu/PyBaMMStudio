# PyBaMM Studio – Streamlit Vertical Slice

An **IDE-style web app** for reproducible battery modeling using [PyBaMM](https://pybamm.org/).  
Built with [Streamlit](https://streamlit.io/), this prototype demonstrates a vertical slice of the planned PyBaMM Studio:

- **Code panel**: edit or paste a PyBaMM script  
- **Parameter panel**: load, search, and edit built-in parameter sets (e.g. Chen2020, Marquis2019, Ai2020, OKane2022, …)  
- **Visualization**: run a model end-to-end and plot voltage vs. time  
- **Repro-Pack export**: download a text or zip bundle containing script, parameters, metadata, and plots  

---

## 🚀 Quick Start

### Run locally

```bash
# 1. Create and activate a Python 3.10–3.12 environment
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit
streamlit run app.py
```
### Run on Streamlit Cloud
https://pybammstudio-dzkyrwzsvsf27fxwqnmfq5.streamlit.app/

## 📦 Features
1. Edit Python code directly in the browser (starter Single Particle Model included).
<img width="2041" height="1740" alt="image" src="https://github.com/user-attachments/assets/86b3a74a-e179-41b7-92e6-b58d63265a45" />

2. Load and inspect built-in parameter sets (Chen2020, Marquis2019, Ai2020, OKane2022, …).
<img width="2041" height="1333" alt="image" src="https://github.com/user-attachments/assets/eb5b3c28-15a9-4a45-8429-45570f445052" />

3. Search & filter parameters, double-click to edit values.
<img width="1828" height="564" alt="image" src="https://github.com/user-attachments/assets/3938c28c-a7f7-4ea8-909a-e975a70d7842" />

4. Run models in-browser with PyBaMM + JAX.
<img width="2041" height="1518" alt="image" src="https://github.com/user-attachments/assets/dca60b6f-606f-43ff-89c3-58f84d473d8b" />

5. Export reproducibility bundles as .txt or .zip files.
<img width="1828" height="337" alt="image" src="https://github.com/user-attachments/assets/d6213f5c-f60d-4d89-bd2b-819b38870c94" />
