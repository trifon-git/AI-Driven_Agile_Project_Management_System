import streamlit as st
import pandas as pd
import os
import numpy as np

INPUT_DIR = "OUTPUT/eval_candidates"
OUTPUT_DIR = "OUTPUT/eval_project_managment"
GNN_IN = os.path.join(INPUT_DIR, "decomposition_eval_gnn.csv")
TRAD_IN = os.path.join(INPUT_DIR, "decomposition_eval_traditional.csv")
GNN_OUT = os.path.join(OUTPUT_DIR, "decomposition_eval_gnn.csv")
TRAD_OUT = os.path.join(OUTPUT_DIR, "decomposition_eval_traditional.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    if os.path.exists(GNN_OUT):
        gnn_df = pd.read_csv(GNN_OUT)
    else:
        gnn_df = pd.read_csv(GNN_IN)
        
    if os.path.exists(TRAD_OUT):
        trad_df = pd.read_csv(TRAD_OUT)
    else:
        trad_df = pd.read_csv(TRAD_IN)
    
    # Ensure columns used for strings/evaluation are initialized as objects (strings) 
    # to avoid "FutureWarning" when empty strings are added to float columns
    str_cols = ['Comments', 'PM Ground Truth Assignee', 'PM Score (1-5)', 'Completeness (1-5)', 'Assignment Score (1-5)', 'Split Comments']
    for col in str_cols:
        if col not in gnn_df.columns:
            gnn_df[col] = None
        if col not in trad_df.columns:
            trad_df[col] = None
            
        gnn_df[col] = gnn_df[col].astype(object)
        trad_df[col] = trad_df[col].astype(object)
            
    return gnn_df, trad_df

def save_data(gnn_df, trad_df):
    gnn_df.to_csv(GNN_OUT, index=False)
    trad_df.to_csv(TRAD_OUT, index=False)
    st.toast("Progress saved successfully!", icon="✅")

def go_to_allocations():
    st.session_state['eval_step'] = "2. Resource Allocation"
    st.session_state['task_inner_idx'] = 0
    save_data(st.session_state.gnn_df, st.session_state.trad_df)

def next_task():
    st.session_state['task_inner_idx'] += 1
    save_data(st.session_state.gnn_df, st.session_state.trad_df)

def finish_project(projects):
    save_data(st.session_state.gnn_df, st.session_state.trad_df)
    if st.session_state['project_idx'] < len(projects) - 1:
        st.session_state['project_idx'] += 1
        st.session_state['eval_step'] = "1. Task Split (Decomposition)"
        st.session_state['task_inner_idx'] = 0
    else:
        st.session_state['all_done'] = True

st.set_page_config(layout="wide", page_title="PM Evaluation App")
st.title("Project Manager Evaluation Tool")

if 'all_done' in st.session_state and st.session_state['all_done']:
    st.balloons()
    st.success("🎉 All project/model combinations evaluated! Thank you.")
    if st.button("Start Over"):
        st.session_state['all_done'] = False
        st.session_state['project_idx'] = 0
        st.session_state['eval_step'] = "1. Task Split (Decomposition)"
        st.rerun()
    st.stop()

# Initialize session state for the dataframes
if 'gnn_df' not in st.session_state or 'trad_df' not in st.session_state:
    st.session_state.gnn_df, st.session_state.trad_df = load_data()

gnn_df = st.session_state.gnn_df
trad_df = st.session_state.trad_df

# Global Progress Tracking across all projects
def get_global_progress(gnn_df, trad_df):
    total_decomp = len(gnn_df)
    comp_decomp = gnn_df['PM Score (1-5)'].apply(lambda x: pd.notna(x) and x != "").sum()
    
    total_alloc_gnn = len(gnn_df)
    comp_alloc_gnn = gnn_df.apply(
        lambda row: pd.notna(row['Assignment Score (1-5)']) and row['Assignment Score (1-5)'] != "" and pd.notna(row['PM Ground Truth Assignee']) and str(row['PM Ground Truth Assignee']).strip() != "",
        axis=1
    ).sum()
    
    total_alloc_trad = len(trad_df)
    comp_alloc_trad = trad_df.apply(
        lambda row: pd.notna(row['Assignment Score (1-5)']) and row['Assignment Score (1-5)'] != "" and pd.notna(row['PM Ground Truth Assignee']) and str(row['PM Ground Truth Assignee']).strip() != "",
        axis=1
    ).sum()
    
    return comp_decomp, total_decomp, (comp_alloc_gnn + comp_alloc_trad), (total_alloc_gnn + total_alloc_trad)

g_comp_decomp, g_tot_decomp, g_comp_alloc, g_tot_alloc = get_global_progress(gnn_df, trad_df)

st.sidebar.header("Navigation")
st.sidebar.subheader("Global Progress")
st.sidebar.progress(g_comp_decomp / g_tot_decomp if g_tot_decomp > 0 else 0, text=f"Task Splits: {g_comp_decomp} / {g_tot_decomp}")
st.sidebar.progress(g_comp_alloc / g_tot_alloc if g_tot_alloc > 0 else 0, text=f"Allocations: {g_comp_alloc} / {g_tot_alloc}")
st.sidebar.divider()

project_models = list(gnn_df[['Project Key', 'Model']].drop_duplicates().itertuples(index=False, name=None))

def get_first_incomplete_project_idx():
    for i, (proj, model) in enumerate(project_models):
        p_gnn = gnn_df[(gnn_df['Project Key'] == proj) & (gnn_df['Model'] == model)]
        p_trad = trad_df[(trad_df['Project Key'] == proj) & (trad_df['Model'] == model)]
        
        # Check GNN splits
        gnn_split_inc = p_gnn['PM Score (1-5)'].isna().any() or p_gnn['Completeness (1-5)'].isna().any()
        
        # Check GNN allocations
        gnn_alloc_inc = p_gnn['Assignment Score (1-5)'].isna().any() or (p_gnn['PM Ground Truth Assignee'].apply(lambda x: pd.isna(x) or str(x).strip() == "")).any()
        
        # Check TRAD allocations (ignoring if traditional missing for this model combo, handle gracefully)
        if not p_trad.empty:
            trad_alloc_inc = p_trad['Assignment Score (1-5)'].isna().any() or (p_trad['PM Ground Truth Assignee'].apply(lambda x: pd.isna(x) or str(x).strip() == "")).any()
        else:
            trad_alloc_inc = False
        
        if gnn_split_inc or gnn_alloc_inc or trad_alloc_inc:
            return i
    return 0

if 'project_idx' not in st.session_state:
    st.session_state['project_idx'] = get_first_incomplete_project_idx()

selected_project, selected_model = project_models[st.session_state['project_idx']]
st.sidebar.info(f"Current Project: **{selected_project}**\n\nModel: **{selected_model}**")

proj_gnn = gnn_df[(gnn_df['Project Key'] == selected_project) & (gnn_df['Model'] == selected_model)]
model_name = selected_model

def get_int_val(val, default):
    if pd.isna(val) or val == "":
        return default
    try:
        return int(float(val))
    except:
        return default

def get_str_val(val):
    return "" if pd.isna(val) else str(val)

# Get a sorted list of unique assignees across the entire dataset to use for the dropdown
unique_assignees = set()
if 'AI Assignee' in gnn_df.columns:
    unique_assignees.update(gnn_df['AI Assignee'].dropna().unique())
if 'AI Assignee' in trad_df.columns:
    unique_assignees.update(trad_df['AI Assignee'].dropna().unique())
if 'PM Ground Truth Assignee' in gnn_df.columns:
    unique_assignees.update(gnn_df['PM Ground Truth Assignee'].dropna().unique())
if 'PM Ground Truth Assignee' in trad_df.columns:
    unique_assignees.update(trad_df['PM Ground Truth Assignee'].dropna().unique())
    
unique_assignees = sorted(list(unique_assignees))
if "Unassigned" not in unique_assignees:
    unique_assignees.insert(0, "Unassigned")
if "" not in unique_assignees:
    unique_assignees.insert(0, "") # Blank option by default

st.header(f"Full Evaluation: {selected_project}")
st.subheader(f"Task Generation Model: {model_name}")
st.info(f"**Project Description:** {proj_gnn.iloc[0].get('Project Description', 'N/A')}")

with st.expander("📊 Total Hours per Assignee", expanded=True):
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0
            
    # GNN Aggregation
    gnn_agg = proj_gnn.copy()
    gnn_agg['Predicted Hours'] = gnn_agg['Predicted Hours'].apply(safe_float)
    gnn_hours = gnn_agg.groupby('AI Assignee')['Predicted Hours'].sum().reset_index()
    gnn_hours.rename(columns={'Predicted Hours': 'GNN Hours'}, inplace=True)
    
    # Traditional Aggregation
    trad_rows = []
    for t_id in proj_gnn['Task ID'].unique():
        t_match = trad_df[(trad_df['Project Key'] == selected_project) & (trad_df['Model'] == selected_model) & (trad_df['Task ID'] == t_id)]
        if t_match.empty:
            t_match = trad_df[(trad_df['Project Key'] == selected_project) & (trad_df['Task ID'] == t_id)]
        if not t_match.empty:
            trad_rows.append(t_match.iloc[0])
            
    if trad_rows:
        trad_agg = pd.DataFrame(trad_rows)
        trad_agg['Predicted Hours'] = trad_agg['Predicted Hours'].apply(safe_float)
        trad_hours = trad_agg.groupby('AI Assignee')['Predicted Hours'].sum().reset_index()
        trad_hours.rename(columns={'Predicted Hours': 'Traditional Hours'}, inplace=True)
    else:
        trad_hours = pd.DataFrame(columns=['AI Assignee', 'Traditional Hours'])
        
    summary_df = pd.merge(trad_hours, gnn_hours, on='AI Assignee', how='outer').fillna(0)
    summary_df.rename(columns={'AI Assignee': 'Assignee'}, inplace=True)
    
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.write("No allocation data available for this project.")

with st.container(border=True):
    st.markdown("### 🏛️ Step 0: Overall Project Split Completeness")
    st.write(":blue[*Did the model break down the project into all the necessary tasks, or are there missing pieces?*]")
    proj_comp_s = st.radio("Project Completeness Score (1-5) *[1=Many missing tasks, 5=Fully covered]*", [1, 2, 3, 4, 5], index=max(0, get_int_val(proj_gnn.iloc[0].get('Completeness (1-5)'), 3) - 1), key=f"ds_proj_comp_{selected_project}_{selected_model}", horizontal=True)

st.divider()

st.markdown("### 📋 Task Breakdown")
st.write("Expand each task below to evaluate both the **Task Split** and the **Resource Allocations**.")

def render_alloc_col(col, model_label, row, df_name, idx, s_task):
    with col:
        st.markdown(f"#### 🤖 {model_label} Model")
        
        # Color coding Based on model type
        if model_label == "GNN":
            st.success(f"**Assignee:** {row.get('AI Assignee', 'N/A')}\n\n**Predicted Hours:** {row.get('Predicted Hours', 'N/A')}")
        else:
            st.info(f"**Assignee:** {row.get('AI Assignee', 'N/A')}\n\n**Predicted Hours:** {row.get('Predicted Hours', 'N/A')}")
        
        st.markdown("**How good is this specific assignment?**")
        ass_score = st.radio("Assignment Score (1-5)", [1, 2, 3, 4, 5], index=max(0, get_int_val(row['Assignment Score (1-5)'], 3) - 1), key=f"al_ass_{selected_project}_{selected_model}_{model_label}_{s_task}", horizontal=True)
        
        st.markdown("**Comments on this assignment?**")
        comments = st.text_area(f"{model_label} Comments", get_str_val(row['Comments']), key=f"al_com_{selected_project}_{selected_model}_{model_label}_{s_task}", height=80)
        
        # Direct session_state DataFrame manipulation to ensure lock-step updates
        st.session_state[df_name].at[idx, 'Assignment Score (1-5)'] = ass_score
        st.session_state[df_name].at[idx, 'Comments'] = comments

# Add split comments col if missing to prevent overriding allocation comments
if 'Split Comments' not in st.session_state.gnn_df.columns:
    st.session_state.gnn_df['Split Comments'] = ""
    st.session_state.trad_df['Split Comments'] = ""

for task_id in proj_gnn['Task ID'].unique():
    idx_gnn = gnn_df[(gnn_df['Project Key'] == selected_project) & (gnn_df['Model'] == selected_model) & (gnn_df['Task ID'] == task_id)].index[0]
    
    # TRAD logic: Try finding for model, if not find any for that project (trad model might be generic or same)
    trad_matches = trad_df[(trad_df['Project Key'] == selected_project) & (trad_df['Model'] == selected_model) & (trad_df['Task ID'] == task_id)]
    if trad_matches.empty:
        trad_matches = trad_df[(trad_df['Project Key'] == selected_project) & (trad_df['Task ID'] == task_id)]
        
    if not trad_matches.empty:
        idx_trad = trad_matches.index[0]
    else:
        st.error(f"Missing matching traditional task for {task_id}")
        continue
    
    row_gnn = gnn_df.loc[idx_gnn]
    row_trad = trad_df.loc[idx_trad]
    
    split_done = pd.notna(row_gnn['PM Score (1-5)']) and row_gnn['PM Score (1-5)'] != "" and pd.notna(row_gnn['Completeness (1-5)']) and row_gnn['Completeness (1-5)'] != ""
    alloc_gnn_done = pd.notna(row_gnn['Assignment Score (1-5)']) and row_gnn['Assignment Score (1-5)'] != "" and pd.notna(row_gnn['PM Ground Truth Assignee']) and str(row_gnn['PM Ground Truth Assignee']).strip() != ""
    alloc_trad_done = pd.notna(row_trad['Assignment Score (1-5)']) and row_trad['Assignment Score (1-5)'] != "" and pd.notna(row_trad['PM Ground Truth Assignee']) and str(row_trad['PM Ground Truth Assignee']).strip() != ""
    
    is_fully_completed = split_done and alloc_gnn_done and alloc_trad_done
    icon = "✅" if is_fully_completed else "⭕"
    
    with st.expander(f"Task {task_id}: {row_gnn['Summary']}", expanded=False, icon=icon):
        # --- TASK SPLIT ---
        with st.container(border=True):
            st.markdown("### 📝 Step 1: Task Split Evaluation")
            st.write(":orange[*Does this task make sense as a chunk of work?*]")
            st.markdown(f"**Description:**\n{row_gnn['Description']}")
            pm_s = st.radio("PM Score (1-5) *[1=Poor, 5=Excellent]*", [1, 2, 3, 4, 5], index=max(0, get_int_val(row_gnn['PM Score (1-5)'], 3) - 1), key=f"ds_pm_{selected_project}_{selected_model}_{task_id}", horizontal=True)
            
            split_comments = st.text_area("Task Split Comments (Optional)", get_str_val(row_gnn.get('Split Comments', '')), key=f"ds_split_com_{selected_project}_{selected_model}_{task_id}", placeholder="Is this task too big/small?", height=80)
        
        st.session_state.gnn_df.at[idx_gnn, 'PM Score (1-5)'] = pm_s
        st.session_state.trad_df.at[idx_trad, 'PM Score (1-5)'] = pm_s
        st.session_state.gnn_df.at[idx_gnn, 'Completeness (1-5)'] = proj_comp_s
        st.session_state.trad_df.at[idx_trad, 'Completeness (1-5)'] = proj_comp_s
        st.session_state.gnn_df.at[idx_gnn, 'Split Comments'] = split_comments
        st.session_state.trad_df.at[idx_trad, 'Split Comments'] = split_comments

        st.divider()
        
        # --- ALLOCATION ---
        with st.container(border=True):
            st.markdown("### 👥 Step 2: Resource Allocation")
            
            # Ground Truth directly under Step 2 (single shared input for the task)
            st.markdown("**Who would you *ideally* assign this to?** (Ground Truth)")
            
            # Determine current value (prefer GNN, fallback to traditional)
            current_gt = get_str_val(row_gnn.get('PM Ground Truth Assignee', ''))
            if not current_gt:
                current_gt = get_str_val(row_trad.get('PM Ground Truth Assignee', ''))
                
            gt_index = unique_assignees.index(current_gt) if current_gt in unique_assignees else 0
            gt_assignee = st.selectbox("Ideal Assignee", unique_assignees, index=gt_index, key=f"gt_shared_{selected_project}_{selected_model}_{task_id}")

            # Update both dataframes exactly the same
            st.session_state.gnn_df.at[idx_gnn, 'PM Ground Truth Assignee'] = gt_assignee
            st.session_state.trad_df.at[idx_trad, 'PM Ground Truth Assignee'] = gt_assignee
            
            st.write(":grey[**Comparison:** Compare how the Traditional model vs the GNN model assigned this specific task.]")
            ac1, _, ac2 = st.columns([1, 0.1, 1])
            
            render_alloc_col(ac1, "Traditional", row_trad, "trad_df", idx_trad, task_id)
            render_alloc_col(ac2, "GNN", row_gnn, "gnn_df", idx_gnn, task_id)

st.divider()
col_prev, col_next = st.columns([1, 1])

with col_prev:
    if st.button("← Previous Project", disabled=st.session_state['project_idx'] == 0, key="btn_prev_project"):
        save_data(st.session_state.gnn_df, st.session_state.trad_df)
        if st.session_state['project_idx'] > 0:
            st.session_state['project_idx'] -= 1
            st.rerun()

with col_next:
    if st.button("Finish Project & Go to Next Project →", type="primary", key="btn_finish_project_combined"):
        save_data(st.session_state.gnn_df, st.session_state.trad_df)
        if st.session_state['project_idx'] < len(project_models) - 1:
            st.session_state['project_idx'] += 1
            st.rerun()
        else:
            st.session_state['all_done'] = True
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("💾 Save Progress", type="secondary", use_container_width=True):
    save_data(st.session_state.gnn_df, st.session_state.trad_df)
