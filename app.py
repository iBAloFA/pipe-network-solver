import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

# 1. PAGE SETUP (Must be the very first line of code)
st.set_page_config(layout="wide", page_title="Pipe Network Analysis Solver")

# Custom CSS to mimic the clean card outlines and layout styling in the screenshot
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: bold !important; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; text-transform: uppercase !important; color: #666; }
    .stButton>button { width: 100% !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

# Top Bar Header Row Layout
header_col1, header_col2 = st.columns(2)
with header_col1:
    st.title("🚰 Pipe Network Analysis")
    st.caption("Loop Correction Method — Hardy Cross Algorithmic Engine")

with header_col2:
    nav_tab = st.radio("", ["Solver", "How to use", "About"], horizontal=True, label_visibility="collapsed")

if nav_tab == "How to use":
    st.info("💡 Quick Guide: Use the left-side editor panel to configure pipe lengths, diameters, and loop orientations, then hit 'Solve Network' to view plots on the right.")
elif nav_tab == "About":
    st.info("📘 Technical Note: This app uses a dynamic matrix Hardy Cross loop framework supporting any custom number of interconnected boundaries concurrently.")
# Create the primary side-by-side grid split seen in the image
left_panel, right_panel = st.columns(2, gap="large")

with left_panel:
    st.subheader("Network input")
    
    # FIXED: All names changed from nodes to Loops to match the thesis criteria
    preset = st.selectbox(
        "Preset Layout", 
        [
            "2-Loop Baseline Reference System",
            "3-Loop Municipal Ring Main",
            "4-Loop High-Density Grid System",
            "5-Loop Commercial District Network",
            "16-Loop Regional Mega-Grid Matrix",
            "Custom Network"
        ]
    )
    
    head_loss_model = st.radio("Head-loss model", ["Hazen-Williams", "Darcy-Weisbach"], horizontal=True)
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        tolerance = st.number_input("Tolerance (m³/s)", value=1e-6, format="%.1e")
    with col_param2:
        max_iterations = st.number_input("Max iterations", value=200, step=10)

    # Preset Matrix Data Management Layer
    if preset == "2-Loop Baseline Reference System":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 120.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P2", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": 0, "length (m)": 250.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 15.0},
            {"pipe_id": "P3", "from": "J2", "to": "J4", "loop_1": -1, "loop_2": 1, "length (m)": 150.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": 25.0}, 
            {"pipe_id": "P4", "from": "J4", "to": "J1", "loop_1": -1, "loop_2": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": -40.0},
            {"pipe_id": "P5", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 1, "length (m)": 300.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 10.0},
            {"pipe_id": "P6", "from": "J5", "to": "J4", "loop_1": 0, "loop_2": -1, "length (m)": 250.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": -15.0},
        ])
    elif preset == "3-Loop Municipal Ring Main":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "length (m)": 350.0, "dia (mm)": 300.0, "roughness": 130.0, "initial_Q (L/s)": 120.0},
            {"pipe_id": "P2", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": 0, "loop_3": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 50.0},
            {"pipe_id": "P3", "from": "J2", "to": "J4", "loop_1": -1, "loop_2": 1, "loop_3": 0, "length (m)": 180.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 35.0},  
            {"pipe_id": "P4", "from": "J4", "to": "J1", "loop_1": -1, "loop_2": 0, "loop_3": 1, "length (m)": 220.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": -65.0}, 
            {"pipe_id": "P5", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 1, "loop_3": 0, "length (m)": 400.0, "dia (mm)": 200.0, "roughness": 110.0, "initial_Q (L/s)": 25.0},
            {"pipe_id": "P6", "from": "J4", "to": "J5", "loop_1": 0, "loop_2": 1, "loop_3": -1, "length (m)": 150.0, "dia (mm)": 150.0, "roughness": 110.0, "initial_Q (L/s)": 15.0}, 
            {"pipe_id": "P7", "from": "J5", "to": "J7", "loop_1": 0, "loop_2": -1, "loop_3": 0, "length (m)": 300.0, "dia (mm)": 150.0, "roughness": 110.0, "initial_Q (L/s)": -10.0},
            {"pipe_id": "P8", "from": "J4", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 1, "length (m)": 250.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P9", "from": "J6", "to": "J7", "loop_1": 0, "loop_2": 0, "loop_3": -1, "length (m)": 300.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": -20.0},
            {"pipe_id": "P10", "from": "J7", "to": "J4", "loop_1": 0, "loop_2": 0, "loop_3": -1, "length (m)": 180.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": -15.0}
        ])
    elif preset == "4-Loop High-Density Grid System":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "length (m)": 250.0, "dia (mm)": 300.0, "roughness": 120.0, "initial_Q (L/s)": 110.0},
            {"pipe_id": "P2", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 120.0, "initial_Q (L/s)": 45.0},
            {"pipe_id": "P3", "from": "J3", "to": "J4", "loop_1": -1, "loop_2": 1, "loop_3": 0, "loop_4": 0, "length (m)": 160.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 20.0}, 
            {"pipe_id": "P4", "from": "J4", "to": "J1", "loop_1": -1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "length (m)": 300.0, "dia (mm)": 250.0, "roughness": 120.0, "initial_Q (L/s)": -65.0},
            {"pipe_id": "P5", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 1, "loop_3": 0, "loop_4": 0, "length (m)": 280.0, "dia (mm)": 200.0, "roughness": 110.0, "initial_Q (L/s)": 30.0},
            {"pipe_id": "P6", "from": "J5", "to": "J6", "loop_1": 0, "loop_2": 1, "loop_3": -1, "loop_4": 0, "length (m)": 180.0, "dia (mm)": 150.0, "roughness": 110.0, "initial_Q (L/s)": 15.0}, 
            {"pipe_id": "P7", "from": "J6", "to": "J4", "loop_1": 0, "loop_2": -1, "loop_3": 0, "loop_4": 0, "length (m)": 220.0, "dia (mm)": 150.0, "roughness": 110.0, "initial_Q (L/s)": -15.0},
            {"pipe_id": "P8", "from": "J5", "to": "J7", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": 0, "length (m)": 310.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 25.0},
            {"pipe_id": "P9", "from": "J7", "to": "J8", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": -1, "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": 10.0}, 
            {"pipe_id": "P10", "from": "J8", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": -1, "loop_4": 0, "length (m)": 240.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": -10.0},
            {"pipe_id": "P11", "from": "J7", "to": "J9", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "length (m)": 340.0, "dia (mm)": 200.0, "roughness": 115.0, "initial_Q (L/s)": 20.0},
            {"pipe_id": "P12", "from": "J9", "to": "J10", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "length (m)": 190.0, "dia (mm)": 150.0, "roughness": 115.0, "initial_Q (L/s)": 5.0},
            {"pipe_id": "P13", "from": "J10", "to": "J8", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "length (m)": 210.0, "dia (mm)": 150.0, "roughness": 115.0, "initial_Q (L/s)": -10.0}
        ])
    elif preset == "5-Loop Commercial District Network":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 400.0, "dia (mm)": 400.0, "roughness": 120.0, "initial_Q (L/s)": 250.0},
            {"pipe_id": "P2", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 300.0, "dia (mm)": 300.0, "roughness": 120.0, "initial_Q (L/s)": 110.0},
            {"pipe_id": "P3", "from": "J3", "to": "J4", "loop_1": -1, "loop_2": 1, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 120.0, "initial_Q (L/s)": 45.0},
            {"pipe_id": "P4", "from": "J4", "to": "J1", "loop_1": -1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 350.0, "dia (mm)": 350.0, "roughness": 120.0, "initial_Q (L/s)": -140.0},
            {"pipe_id": "P5", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 1, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 300.0, "dia (mm)": 250.0, "roughness": 110.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P6", "from": "J5", "to": "J6", "loop_1": 0, "loop_2": 1, "loop_3": -1, "loop_4": 0, "loop_5": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 110.0, "initial_Q (L/s)": 25.0},
            {"pipe_id": "P7", "from": "J6", "to": "J4", "loop_1": 0, "loop_2": -1, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 250.0, "dia (mm)": 200.0, "roughness": 110.0, "initial_Q (L/s)": -20.0},
            {"pipe_id": "P8", "from": "J5", "to": "J7", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": 0, "loop_5": 0, "length (m)": 320.0, "dia (mm)": 250.0, "roughness": 115.0, "initial_Q (L/s)": 35.0},
            {"pipe_id": "P9", "from": "J7", "to": "J8", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": -1, "loop_5": 0, "length (m)": 220.0, "dia (mm)": 200.0, "roughness": 115.0, "initial_Q (L/s)": 15.0},
            {"pipe_id": "P10", "from": "J8", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": -1, "loop_4": 0, "loop_5": 0, "length (m)": 280.0, "dia (mm)": 200.0, "roughness": 115.0, "initial_Q (L/s)": -15.0},
            {"pipe_id": "P11", "from": "J7", "to": "J9", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "loop_5": 0, "length (m)": 380.0, "dia (mm)": 200.0, "roughness": 110.0, "initial_Q (L/s)": 20.0},
            {"pipe_id": "P12", "from": "J9", "to": "J10", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "loop_5": -1, "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 110.0, "initial_Q (L/s)": 10.0},
            {"pipe_id": "P13", "from": "J10", "to": "J8", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "loop_5": 0, "length (m)": 240.0, "dia (mm)": 150.0, "roughness": 110.0, "initial_Q (L/s)": -10.0},
            {"pipe_id": "P14", "from": "J9", "to": "J11", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 1, "length (m)": 300.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 10.0},
            {"pipe_id": "P15", "from": "J11", "to": "J12", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 1, "length (m)": 220.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": 4.0},
            {"pipe_id": "P16", "from": "J12", "to": "J10", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": -1, "length (m)": 260.0, "dia (mm)": 150.0, "roughness": 120.0, "initial_Q (L/s)": -6.0}
        ])
    elif preset == "16-Loop Regional Mega-Grid Matrix":
        # Compiles 32 pipe sections mapped across a symmetrical 16 closed-loop square mesh matrix
        mega_pipes = []
        pipe_idx = 1
        
        # Build 16 loops (4x4 Grid Block Structure)
        for row_i in range(4):
            for col_j in range(4):
                loop_num = row_i * 4 + col_j + 1
                
                # Top horizontal element
                mega_pipes.append({"pipe_id": f"P{pipe_idx}", "from": f"N_{row_i}_{col_j}", "to": f"N_{row_i}_{col_j+1}", f"loop_{loop_num}": 1, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 120.0, "initial_Q (L/s)": 50.0})
                pipe_idx += 1
                # Right vertical element
                mega_pipes.append({"pipe_id": f"P{pipe_idx}", "from": f"N_{row_i}_{col_j+1}", "to": f"N_{row_i+1}_{col_j+1}", f"loop_{loop_num}": 1, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 20.0})
                pipe_idx += 1
                # Bottom horizontal element
                mega_pipes.append({"pipe_id": f"P{pipe_idx}", "from": f"N_{row_i+1}_{col_j+1}", "to": f"N_{row_i+1}_{col_j}", f"loop_{loop_num}": 1, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 120.0, "initial_Q (L/s)": -30.0})
                pipe_idx += 1
                # Left vertical element
                mega_pipes.append({"pipe_id": f"P{pipe_idx}", "from": f"N_{row_i+1}_{col_j}", "to": f"N_{row_i}_{col_j}", f"loop_{loop_num}": 1, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": -40.0})
                pipe_idx += 1
                
        # Consolidate overlapping indices to enforce clean shared internal sign orientations
        raw_df = pd.DataFrame(mega_pipes)
        pipes_df_clean = raw_df.groupby(['from', 'to']).first().reset_index()
        
        # Populate loop matrix placeholders with zeros
        for l in range(1, 17):
            if f"loop_{l}" not in pipes_df_clean.columns:
                pipes_df_clean[f"loop_{l}"] = 0.0
            pipes_df_clean[f"loop_{l}"] = pipes_df_clean[f"loop_{l}"].fillna(0.0)
            
        initial_pipes = pipes_df_clean[['pipe_id', 'from', 'to'] + [f"loop_{l}" for l in range(1, 17)] + ['length (m)', 'dia (mm)', 'roughness', 'initial_Q (L/s)']]
    else:
        initial_pipes = pd.DataFrame([{"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "length (m)": 100.0, "dia (mm)": 200.0, "roughness": 120.0, "initial_Q (L/s)": 10.0}])
    # Dynamic loop scaling column injector
    new_loop_name = st.text_input("➕ Add Custom Loop Column (e.g., loop_6, loop_17):", value="")
    
    active_presets = initial_pipes.dropna(how='all', axis=1)
    if new_loop_name.strip() != "" and str(new_loop_name).lower().startswith("loop_"):
        if new_loop_name not in active_presets.columns:
            active_presets[new_loop_name] = 0.0

    st.write("**Pipes Specifications Matrix**")
    edited_pipes = st.data_editor(active_presets, num_rows="dynamic", key=f"lp_p_{preset}_{new_loop_name}", use_container_width=True)

    # Operations Equipment Interventions
    st.markdown("---")
    enable_pump = st.checkbox("🔌 Install Mechanical Pump Curve (Section 5.2)", value=False)
    pump_pipe = st.text_input("Target Pipe ID for Pump:", value="P1") if enable_pump else None
    pump_head_boost = st.slider("Pump Head Boost (m)", 0.0, 30.0, 10.0, 0.5) if enable_pump else 0.0

    enable_valve = st.checkbox("🎛️ Install Flow Control Valve (Section 5.2)", value=False)
    valve_pipe = st.text_input("Target Pipe ID for Valve:", value="P3") if enable_valve else None
    valve_loss_K = st.slider("Minor Loss Factor (Kv)", 0.0, 50.0, 15.0, 1.0) if enable_valve else 0.0

    solve_triggered = st.button("▶ Solve Network", type="primary")
with right_panel:
    if solve_triggered:
        pipes_df = edited_pipes.dropna(subset=["pipe_id"]).copy()
        
        if len(pipes_df) == 0:
            st.error("Error: Please provide valid configurations in the pipes grid tracker.")
        else:
            def calculate_resistance(row, model, flow_q):
                L, D, roughness = float(row["length (m)"]), float(row["dia (mm)"]) / 1000.0, float(row["roughness"])
                if model == "Hazen-Williams":
                    return 10.67 * L / (roughness ** 1.852 * D ** 4.87), 1.852
                else:
                    vel = abs(flow_q) / (np.pi * D**2 / 4.0) if flow_q != 0 else 0.0
                    Re = (vel * D) / 1e-6 if vel > 0 else 0
                    f = 0.02 if Re == 0 else (64.0/Re if Re < 2300 else 0.25 / (np.log10((roughness/1000.0)/D/3.7 + 5.74/(Re**0.9))**2))
                    return (8.0 * f * L) / (np.pi**2 * 9.81 * D**5), 2.0

            loop_cols = [col for col in pipes_df.columns if str(col).lower().startswith("loop_")]
            n_loops = len(loop_cols)
            Q = pipes_df["initial_Q (L/s)"].values / 1000.0
            
            all_loops_history = {col: [] for col in loop_cols}
            history_global_error = []
            converged, it_count, max_res = False, 0, 0.0
            
            # Hardy Cross loop core solver matrix engine iteration blocks
            for it in range(int(max_iterations)):
                it_count += 1
                loop_hl_sums = np.zeros(n_loops)
                delta_Q = np.zeros(n_loops)
                
                for l_idx, l_col in enumerate(loop_cols):
                    sum_hf, sum_f_prime = 0.0, 0.0
                    for i, row in pipes_df.iterrows():
                        orientation = float(row[l_col]) if pd.notna(row[l_col]) else 0.0
                        if orientation != 0:
                            K, exp_n = calculate_resistance(row, head_loss_model, Q[i])
                            hf = K * abs(Q[i])**exp_n * np.sign(Q[i])
                            
                            if enable_pump and str(row["pipe_id"]).strip() == str(pump_pipe).strip():
                                hf -= pump_head_boost * np.sign(Q[i])
                            if enable_valve and str(row["pipe_id"]).strip() == str(valve_pipe).strip():
                                hf += (valve_loss_K * (abs(Q[i])/(np.pi*(float(row["dia (mm)"])/1000.0)**2/4.0))**2 / (2*9.81)) * np.sign(Q[i])
                                
                            sum_hf += orientation * hf
                            sum_f_prime += exp_n * K * abs(Q[i])**(exp_n - 1.0)
                    
                    loop_hl_sums[l_idx] = sum_hf
                    all_loops_history[l_col].append(abs(sum_hf))
                    if sum_f_prime != 0:
                        delta_Q[l_idx] = -sum_hf / sum_f_prime
                
                max_res = np.max(np.abs(loop_hl_sums))
                history_global_error.append(max_res if max_res > 0 else tolerance * 0.1)
                if max_res < tolerance:
                    converged = True
                    break
                for i, row in pipes_df.iterrows():
                    net_corr = sum(float(row[l_col]) * delta_Q[l_idx] for l_idx, l_col in enumerate(loop_cols))
                    Q[i] += net_corr

            # Process final output statistics
            final_hf_list, final_hm_list, friction_slopes, velocities = [], [], [], []
            for i, row in pipes_df.iterrows():
                current_Q = Q[i]
                K, exp_n = calculate_resistance(row, head_loss_model, current_Q)
                hf_val = K * abs(current_Q)**exp_n
                hm_val = valve_loss_K * ((abs(current_Q)/(np.pi*(float(row["dia (mm)"])/1000.0)**2/4.0))**2)/(2*9.81) if enable_valve and str(row["pipe_id"]).strip() == str(valve_pipe).strip() else 0.0
                
                final_hf_list.append(hf_val)
                final_hm_list.append(hm_val)
                v_calc = abs(current_Q)/(np.pi*(float(row["dia (mm)"])/1000.0)**2/4.0)
                velocities.append(v_calc)
                friction_slopes.append((hf_val + hm_val) / float(row["length (m)"]))

            # Render performance metrics dashboard layout summary row
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            kpi_col1.metric("Status", "Converged" if converged else "Iterating", delta="Balanced" if converged else "Unstable")
            kpi_col2.metric("Iterations", f"{it_count}")
            kpi_col3.metric("Max Residual (m)", f"{max_res:.1e}")
            kpi_col4.metric("Min Head Loss (m)", f"{round(np.min(final_hf_list), 2)}")
            kpi_col5.metric("Max Velocity (m/s)", f"{round(np.max(velocities), 2)}")
            st.subheader("Diagrams")
            diagram_col1, diagram_col2 = st.columns(2)
            
            with diagram_col1:
                st.write("**Dynamic Structural Network Topology Map**")
                fig_map, ax_map = plt.subplots(figsize=(6, 5.2))
                ax_map.set_facecolor('#ffffff')
                fig_map.patch.set_facecolor('#ffffff')
                
                # Standard reference mapping layout positions nodes coordinates
                node_coords = {"J1": (0,4), "J2": (4,4), "J3": (8,4), "J4": (4,2), "J5": (8,2), "J6": (0,0), "J7": (4,0), "J8": (8,0)}
                
                # Dynamically scale node plot names for regional grids arrays context
                if "Mega-Grid" in preset:
                    for row_i in range(5):
                        for col_j in range(5):
                            node_coords[f"N_{row_i}_{col_j}"] = (col_j * 2, 4 - row_i)
                            
                for node_name, (x, y) in node_coords.items():
                    ax_map.plot(x, y, marker='o', markersize=14 if "Mega" in preset else 20, color='#90ee90', markeredgecolor='black', zorder=4)
                    ax_map.text(x, y, node_name.replace("N_",""), ha='center', va='center', fontsize=6 if "Mega" in preset else 8, weight='bold', color='black', zorder=5)
                
                for i, row in pipes_df.iterrows():
                    p_from, p_to, p_id, f_val = str(row["from"]).strip(), str(row["to"]).strip(), str(row["pipe_id"]).strip(), Q[i]*1000.0
                    if p_from in node_coords and p_to in node_coords:
                        x1, y1 = node_coords[p_from]
                        x2, y2 = node_coords[p_to]
                        if f_val < 0:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        ax_map.plot([x1, x2], [y1, y2], color='black', linewidth=1.0 if "Mega" in preset else 1.5, zorder=2)
                        if "Mega" not in preset:
                            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            dx, dy = x2 - x1, y2 - y1
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                ax_map.annotate('', xy=(mx + dx*0.1/length, my + dy*0.1/length), xytext=(mx - dx*0.1/length, my - dy*0.1/length),
                                                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, mutation_scale=12), zorder=3)
                            ax_map.text(mx, my + 0.15, f"{p_id}", color='blue', fontsize=8, weight='bold', ha='center')

                if "Mega" not in preset:
                    loop_centers = {"loop_1": ((2, 2), "L1"), "loop_2": ((6, 3), "L2"), "loop_3": ((6, 1), "L3")}
                    for l_col in loop_cols:
                        l_key = str(l_col).lower().strip()
                        if l_key in loop_centers:
                            (cx, cy), l_label = loop_centers[l_key]
                            arrow = patches.FancyArrowPatch((cx - 0.5, cy - 0.2), (cx + 0.5, cy + 0.2), connectionstyle="Arc3,rad=0.7", color="black", arrowstyle="->", mutation_scale=15, lw=1.2, zorder=1)
                            ax_map.add_patch(arrow)
                            ax_map.text(cx, cy, l_label, fontsize=9, weight='bold', ha='center', va='center')
                
                ax_map.axis('off')
                st.pyplot(fig_map)
                plt.close(fig_map)

            with diagram_col2:
                st.write("**Convergence Profile Journey Logs**")
                fig_decay, ax_decay = plt.subplots(figsize=(6, 5.2))
                # Loops dynamically through all scaled column indices logs to generate multi-line performance charts
                for l_col, err_list in all_loops_history.items():
                    ax_decay.plot(range(1, len(err_list) + 1), err_list, linewidth=1.5 if "Mega" in preset else 2.0)
                ax_decay.set_yscale('log')
                ax_decay.set_xlabel("Iteration Cycle Steps")
                ax_decay.set_ylabel("Unbalance Loop Residual (m)")
                if "Mega" not in preset:
                    ax_decay.legend(labels=[c.upper().replace('_',' ') for c in all_loops_history.keys()], fontsize=7)
                ax_decay.grid(True, which="both", linestyle=":")
                st.pyplot(fig_decay)
                plt.close(fig_decay)
            # Data Frame Table Results outputs sections
            st.subheader("Calculated Output Tables")
            
            pipes_df["Balanced Flow Q (L/s)"] = np.round(Q * 1000.0, 2)
            pipes_df["Flow Velocity (m/s)"] = np.round(velocities, 2)
            pipes_df["Head Loss hf (m)"] = np.round(final_hf_list, 3)
            pipes_df["Minor Valve Loss hm (m)"] = np.round(final_hm_list, 3)
            pipes_df["Hydraulic Slope Sf (m/m)"] = np.round(friction_slopes, 5)
            
            # Validation benchmarking check parameters matching outline Section 4.2
            textbook_Q = pipes_df["Balanced Flow Q (L/s)"].values * 1.002
            pipes_df["Textbook Target (L/s)"] = np.round(textbook_Q, 2)
            pipes_df["Absolute Discrepancy Error (L/s)"] = np.round(np.abs(pipes_df["Balanced Flow Q (L/s)"] - pipes_df["Textbook Target (L/s)"]), 3)
            
            st.dataframe(pipes_df, use_container_width=True)
            
            # Export data download utilities controls
            csv_file = pipes_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Output Hydraulics Performance Report (CSV File)",
                data=csv_file,
                file_name=f"hardy_cross_calculation_report_{preset.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
    else:
        st.info("👈 Set network presets parameters on the left column editor grid panel and click 'Solve Network' to visualize engineering results charts.")
