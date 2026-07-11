import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 1. PAGE SETUP (Must be the very first line of code)
st.set_page_config(layout="wide", page_title="Pipe Network Solver")

st.title("🚰 Pipe Network Analysis Solver")
st.caption("Nodal Head Correction Method (Newton-Raphson Engine)")

# ==========================================
# ABOUT & HOW TO USE INSTRUCTIONS GUIDE
# ==========================================
with st.expander("❓ About & How to Use This Solver", expanded=True):
    col_about, col_guide = st.columns(2)
    with col_about:
        st.markdown("""
        ### 📘 About the Application
        This application is an educational hydraulic simulator designed to analyze steady-state water distribution networks. 
        It utilizes the **Nodal Head Correction Method** powered by a **Newton-Raphson iteration matrix engine** to solve for 
        unknown pipe flow rates and junction pressures simultaneously based on mass continuity and energy conservation laws.
        """)
    with col_guide:
        st.markdown("""
        ### 🚀 How to Use It
        1. **Configure Settings:** Choose a pre-defined layout or a custom option using the dropdown menu in the left sidebar.
        2. **Select Math Models:** Choose between the **Hazen-Williams** or **Darcy-Weisbach** equations to calculate pipe friction losses.
        3. **Edit Grid Data:** Modify elevations, flow demands, lengths, or diameters directly inside the interactive spreadsheet grids below.
        4. **Compute Solutions:** Click the red **"Solve Network"** button to run calculations, view network graphs, and read the automated engineering analysis report.
        """)

# 2. SIDEBAR CONTROLS
st.sidebar.header("🔧 Solver Settings")

preset = st.sidebar.selectbox(
    "Select Network Preset", 
    [
        "Figure 1 Network (4 nodes)", 
        "8-Node Medium Loop Network",
        "Custom Network"
    ]
)

head_loss_model = st.sidebar.radio("Head-Loss Model", ["Hazen-Williams", "Darcy-Weisbach"])
tolerance = st.sidebar.number_input("Convergence Tolerance (m³/s)", value=1e-5, format="%.1e")
max_iterations = st.sidebar.number_input("Max Iterations", value=1000000, step=1000)
# 3. PRESET LOADING LOGIC
if preset == "Figure 1 Network (4 nodes)":
    initial_nodes = pd.DataFrame([
        {"id": "A", "type": "reservoir", "elev. (m)": 50.0, "demand (m³/s)": -0.20, "fixed head (m)": 50.0},
        {"id": "B", "type": "junction", "elev. (m)": 15.0, "demand (m³/s)": 0.05, "fixed head (m)": None},
        {"id": "C", "type": "junction", "elev. (m)": 10.0, "demand (m³/s)": 0.08, "fixed head (m)": None},
        {"id": "D", "type": "junction", "elev. (m)": 12.0, "demand (m³/s)": 0.07, "fixed head (m)": None},
    ])
    initial_pipes = pd.DataFrame([
        {"pipe_id": "P1", "from": "A", "to": "B", "length (m)": 300.0, "dia (mm)": 200.0, "roughness": 130.0},
        {"pipe_id": "P2", "from": "B", "to": "C", "length (m)": 250.0, "dia (mm)": 150.0, "roughness": 130.0},
        {"pipe_id": "P3", "from": "C", "to": "D", "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 130.0},
        {"pipe_id": "P4", "from": "D", "to": "B", "length (m)": 180.0, "dia (mm)": 150.0, "roughness": 130.0},
    ])

elif preset == "8-Node Medium Loop Network":
    initial_nodes = pd.DataFrame([
        {"id": "N1", "type": "reservoir", "elev. (m)": 60.0, "demand (m³/s)": -0.24, "fixed head (m)": 60.0},
        {"id": "N2", "type": "junction", "elev. (m)": 20.0, "demand (m³/s)": 0.04, "fixed head (m)": None},
        {"id": "N3", "type": "junction", "elev. (m)": 18.0, "demand (m³/s)": 0.03, "fixed head (m)": None},
        {"id": "N4", "type": "junction", "elev. (m)": 22.0, "demand (m³/s)": 0.05, "fixed head (m)": None},
        {"id": "N5", "type": "junction", "elev. (m)": 25.0, "demand (m³/s)": 0.02, "fixed head (m)": None},
        {"id": "N6", "type": "reservoir", "elev. (m)": 55.0, "demand (m³/s)": 0.00, "fixed head (m)": 55.0},
        {"id": "N7", "type": "junction", "elev. (m)": 15.0, "demand (m³/s)": 0.06, "fixed head (m)": None},
        {"id": "N8", "type": "junction", "elev. (m)": 17.0, "demand (m³/s)": 0.04, "fixed head (m)": None},
    ])
    initial_pipes = pd.DataFrame([
        {"pipe_id": "P1", "from": "N1", "to": "N2", "length (m)": 400.0, "dia (mm)": 300.0, "roughness": 120.0},
        {"pipe_id": "P2", "from": "N2", "to": "N3", "length (m)": 300.0, "dia (mm)": 200.0, "roughness": 120.0},
        {"pipe_id": "P3", "from": "N3", "to": "N4", "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 120.0},
        {"pipe_id": "P4", "from": "N4", "to": "N1", "length (m)": 350.0, "dia (mm)": 250.0, "roughness": 120.0},
        {"pipe_id": "P5", "from": "N3", "to": "N5", "length (m)": 500.0, "dia (mm)": 150.0, "roughness": 110.0},
        {"pipe_id": "P6", "from": "N5", "to": "N6", "length (m)": 250.0, "dia (mm)": 300.0, "roughness": 120.0},
        {"pipe_id": "P7", "from": "N6", "to": "N7", "length (m)": 450.0, "dia (mm)": 200.0, "roughness": 110.0},
        {"pipe_id": "P8", "from": "N7", "to": "N8", "length (m)": 300.0, "dia (mm)": 150.0, "roughness": 110.0},
        {"pipe_id": "P9", "from": "N8", "to": "N2", "length (m)": 600.0, "dia (mm)": 250.0, "roughness": 120.0},
    ])
else:
    initial_nodes = pd.DataFrame(columns=["id", "type", "elev. (m)", "demand (m³/s)", "fixed head (m)"])
    initial_pipes = pd.DataFrame(columns=["pipe_id", "from", "to", "length (m)", "dia (mm)", "roughness"])

# 4. EDITABLE TABLES SETUP
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Nodes Configuration")
    edited_nodes = st.data_editor(initial_nodes, num_rows="dynamic", key=f"nodes_{preset}", use_container_width=True)

with col2:
    st.subheader("🚀 Pipes Configuration")
    edited_pipes = st.data_editor(initial_pipes, num_rows="dynamic", key=f"pipes_{preset}", use_container_width=True)
# 5. MATHEMATICAL RESISTANCE COMPUTER
def compute_pipe_resistance(row, model):
    L = float(row["length (m)"])
    D = float(row["dia (mm)"]) / 1000.0
    C = float(row["roughness"])
    if model == "Hazen-Williams":
        return 10.67 * L / (C**1.852 * D**4.87)
    else:
        f_friction = 0.02  # Simplified baseline friction factor
        return 8.0 * f_friction * L / (np.pi**2 * 9.81 * D**5)

if st.button("🔴 Solve Network", type="primary"):
    st.write("---")
    
    nodes_df = edited_nodes.dropna(subset=["id"]).copy()
    pipes_df = edited_pipes.dropna(subset=["pipe_id", "from", "to"]).copy()
    
    if len(nodes_df) == 0 or len(pipes_df) == 0:
        st.error("Error: Please make sure both configuration tables contain valid data rows.")
    else:
        with st.spinner("Processing network topologies with active matrix engine..."):
            node_ids = list(nodes_df["id"].unique())
            n_nodes = len(node_ids)
            node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            
            heads = np.zeros(n_nodes)
            fixed_mask = np.zeros(n_nodes, dtype=bool)
            
            for idx, row in nodes_df.iterrows():
                n_i = node_to_idx[row["id"]]
                if str(row["type"]).lower() == "reservoir" and pd.notna(row["fixed head (m)"]):
                    heads[n_i] = float(row["fixed head (m)"])
                    fixed_mask[n_i] = True
                else:
                    heads[n_i] = float(row["elev. (m)"]) + 20.0
            
            exp_n = 1.852 if head_loss_model == "Hazen-Williams" else 2.0
            history_residuals = []
            converged = False
            it_count = 0
            
            for it in range(int(max_iterations)):
                it_count += 1
                node_residuals = np.zeros(n_nodes)
                jacobian = np.zeros((n_nodes, n_nodes))
                
                for _, pipe in pipes_df.iterrows():
                    idx_f = node_to_idx[pipe["from"]]
                    idx_t = node_to_idx[pipe["to"]]
                    K = compute_pipe_resistance(pipe, head_loss_model)
                    dh = heads[idx_f] - heads[idx_t]
                    
                    flow_q = (abs(dh) / K) ** (1.0 / exp_n) * np.sign(dh) if dh != 0 else 0.0
                    node_residuals[idx_f] -= flow_q
                    node_residuals[idx_t] += flow_q
                    
                    dq_dh = (1.0 / (exp_n * K)) * (abs(dh) / K) ** ((1.0 / exp_n) - 1.0) if dh != 0 else 0.0
                    jacobian[idx_f, idx_f] += dq_dh
                    jacobian[idx_f, idx_t] -= dq_dh
                    jacobian[idx_t, idx_f] -= dq_dh
                    jacobian[idx_t, idx_t] += dq_dh

                for idx, row in nodes_df.iterrows():
                    n_i = node_to_idx[row["id"]]
                    if not fixed_mask[n_i]:
                        node_residuals[n_i] -= float(row["demand (m³/s)"])
                
                unknown_residuals = node_residuals[~fixed_mask]
                max_res = np.max(np.abs(unknown_residuals)) if len(unknown_residuals) > 0 else 0.0
                history_residuals.append(max_res if max_res > 0 else tolerance * 0.1)
                
                if max_res < tolerance:
                    converged = True
                    break
                    
                rhs = node_residuals
                for i in range(n_nodes):
                    if fixed_mask[i]:
                        jacobian[i, :] = 0.0
                        jacobian[:, i] = 0.0
                        jacobian[i, i] = 1.0
                        rhs[i] = 0.0
                        
                try:
                    delta_h = np.linalg.solve(jacobian, rhs)
                    heads += delta_h
                except np.linalg.LinAlgError:
                    break

            pipe_flows, pipe_velocities = [], []
            for _, pipe in pipes_df.iterrows():
                idx_f = node_to_idx[pipe["from"]]
                idx_t = node_to_idx[pipe["to"]]
                K = compute_pipe_resistance(pipe, head_loss_model)
                dh = heads[idx_f] - heads[idx_t]
                flow_q = (abs(dh) / K) ** (1.0 / exp_n) * np.sign(dh) if dh != 0 else 0.0
                area = np.pi * (float(pipe["dia (mm)"]) / 1000.0) ** 2 / 4.0
                pipe_flows.append(flow_q * 1000.0)
                pipe_velocities.append(abs(flow_q) / area)

            pressure_heads = heads - nodes_df['elev. (m)'].values
            min_p_head = np.min(pressure_heads)
            max_v = np.max(pipe_velocities)

            # Display KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Status", "Converged" if converged else "Failed")
            kpi2.metric("Iterations", f"{it_count}/{max_iterations}")
            kpi3.metric("Min Pressure Head", f"{round(min_p_head, 2)} m")
            kpi4.metric("Max Velocity", f"{round(max_v, 2)} m/s")

            # ==========================================
            # AUTOMATED EXPLANATION GENERATOR REPORT
            # ==========================================
            st.subheader("💡 Automated Hydraulic Calculation Report")
            with st.container(border=True):
                if converged:
                    st.success(f"✅ **Convergence Achieved:** The solver successfully balanced the system in **{it_count} iterations** using the {head_loss_model} model. Mass constraints at all network junctions are balanced within your set tolerance limits.")
                else:
                    st.error("❌ **Convergence Failed:** The network could not reach a balance within the maximum iteration threshold. Please check for conflicting input parameters or isolated segments.")
                
                if min_p_head < 0:
                    st.warning(f"⚠️ **Negative Pressure Risk Detected ({round(min_p_head, 2)} m):** Your minimum calculated pressure head drops below zero. In real systems, this indicates **suction, vacuum conditions, or cavitation risks**. To fix this, consider increasing your source reservoir height or expanding downstream pipe diameters.")
                elif min_p_head < 15.0:
                    st.info(f"ℹ️ **Low Operational Pressure ({round(min_p_head, 2)} m):** Pressures are stable but lean below typical standard municipal targets (~15m to 20m). Flow delivery might feel weak during peak hours.")
                else:
                    st.success(f"💎 **Healthy System Pressures ({round(min_p_head, 2)} m):** All nodes maintain positive, reliable pressure levels that are well within safe domestic operational standards.")
                
                if max_v > 2.5:
                    st.warning(f"⚡ **High Velocity Alert ({round(max_v, 2)} m/s):** Velocities exceed standard guidelines (usually capped at 2.0-2.5 m/s). This triggers major friction losses and introduces risks of severe **water hammer surge pressures**.")
                elif max_v < 0.3:
                    st.info(f"🐌 **Low Velocity Profile ({round(max_v, 2)} m/s):** Water moves slowly through the largest pipelines. While head losses remain small, very slow water movement (< 0.3 m/s) can lead to sediment settling and stagnation.")
                else:
                    st.success(f"🌊 **Optimal Flow Velocity ({round(max_v, 2)} m/s):** Fluid speeds stay within the sweet spot (0.6m/s - 2.0m/s), keeping water moving without causing unnecessary pipeline wear.")

            # 6. GRAPH GENERATION DASHBOARD
            st.subheader("📊 Simulation Analysis Diagrams")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.write("**Network Map Layout (Section 3.2)**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                G = nx.DiGraph()
                for _, row in pipes_df.iterrows():
                    G.add_edge(row['from'], row['to'], label=row['pipe_id'])
                pos = nx.circular_layout(G) if n_nodes > 5 else nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, font_size=8, font_weight='bold', ax=ax1)
                st.pyplot(fig1)
                plt.close(fig1)

            with chart_col2:
                st.write("**Convergence Residual Tracking (Section 4.3)**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(range(1, len(history_residuals) + 1), history_residuals, marker='o', color='crimson')
                ax2.set_yscale('log')
                ax2.set_xlabel("Iteration Step")
                ax2.set_ylabel("Max Balance Error (m³/s)")
                ax2.grid(True, which="both", linestyle="--")
                st.pyplot(fig2)
                plt.close(fig2)

            # 7. RESULTS DATA FRAME DUMP TABS
            st.subheader("📈 Calculated Performance Tables")
            tab_nodes, tab_pipes = st.tabs(["Nodal Pressures & Heads", "Pipe Flow Characteristics"])
            
            with tab_nodes:
                nodes_df["Calculated Head (m)"] = np.round(heads, 2)
                nodes_df["Pressure Head (m)"] = np.round(pressure_heads, 2)
                st.dataframe(nodes_df, use_container_width=True)
                
            with tab_pipes:
                pipes_df["Flow Rate (L/s)"] = np.round(pipe_flows, 2)
                pipes_df["Velocity (m/s)"] = np.round(pipe_velocities, 2)
                st.dataframe(pipes_df, use_container_width=True)
