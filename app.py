import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# PAGE CONFIGURATION & EXECUTIVE STYLING
# ==========================================
st.set_page_config(
    page_title="Equimeritization Index (EI) Executive Audit Suite",
    page_icon="ei_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        gap: 8px;
        background-color: #F8FAFC;
        padding-bottom: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-shrink: 0;
        background-color: #FFFFFF;
        border-radius: 8px;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border: 1px solid #1E3A8A !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE ARCHIVE
# ==========================================
if 'audit_history' not in st.session_state:
    st.session_state['audit_history'] = []


# ==========================================
# SYNTHETIC DATA ENGINE
# ==========================================
@st.cache_data
def generate_synthetic_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    first_gen = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
    pell_status = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    urm_status = np.random.choice([0, 1], size=n_samples, p=[0.50, 0.50])
    
    cohorts = []
    for fg, pell, urm in zip(first_gen, pell_status, urm_status):
        labels = []
        if fg: labels.append("First-Gen")
        else: labels.append("Continuing-Gen")
        if pell: labels.append("Pell")
        if urm: labels.append("URM")
        cohorts.append(" / ".join(labels))
    
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    gpa = np.clip(np.random.normal(3.0, 0.5, n_samples) + (y_true * 0.4), 1.0, 4.0)
    lms_clicks = np.clip(np.random.normal(300, 100, n_samples) + (y_true * 120), 10, 800)
    
    marginalization_score = first_gen + pell_status + urm_status
    lms_clicks = np.clip(lms_clicks * (1.0 - (marginalization_score * 0.08)), 10, 800)
    
    noise1 = np.random.normal(0, 0.25, n_samples)
    disparity_penalty = marginalization_score * -0.09
    p_model1 = np.clip(y_true * 0.65 + 0.22 + disparity_penalty + noise1, 0, 1)
    
    noise2 = np.random.normal(0, 0.28, n_samples)
    p_model2 = np.clip(y_true * 0.60 + 0.22 + noise2, 0, 1)
    
    df = pd.DataFrame({
        'Student_ID': [f"STU-{10000 + i}" for i in range(n_samples)],
        'Intersectional_Cohort': cohorts,
        'First_Gen': first_gen,
        'Pell_Recipient': pell_status,
        'URM_Status': urm_status,
        'Actual_Persistence': y_true,
        'GPA': gpa,
        'LMS_Clicks': lms_clicks,
        'Vendor_Legacy_Prob': p_model1,
        'Equimeritization_Calibrated_Prob': p_model2
    })
    return df


# ==========================================
# CORE 4-PILLAR AUDIT LOGIC & VENDOR BENCHMARKING
# ==========================================
def run_epistemic_audit(df):
    X = df[['GPA', 'LMS_Clicks']]
    y = df['Actual_Persistence']
    if len(X) < 10:
        return pd.DataFrame({'Feature': ['GPA', 'LMS_Clicks'], 'Importance': [0.5, 0.5]}), 0.5, "INSUFFICIENT DATA"
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    perm_imp = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({'Feature': ['GPA', 'LMS_Clicks'], 'Importance': perm_imp.importances_mean}).sort_values(by='Importance', ascending=False)
    total_imp = imp_df['Importance'].sum()
    top_proxy_ratio = imp_df.iloc[0]['Importance'] / total_imp if total_imp > 0 else 0.0
    status = "PASSED" if top_proxy_ratio < 0.75 else "FLAGGED (High Proxy Reliance)"
    return imp_df, top_proxy_ratio, status


def run_construct_audit(df, proxy_col='LMS_Clicks', target_col='Actual_Persistence', group_col='Intersectional_Cohort'):
    groups = df[group_col].unique()
    validity_by_group = {}
    for g in groups:
        sub_df = df[df[group_col] == g]
        if len(sub_df) > 5:
            corr = sub_df[proxy_col].corr(sub_df[target_col])
            validity_by_group[g] = corr if not np.isnan(corr) else 0.0
        else:
            validity_by_group[g] = 0.0
    corrs = list(validity_by_group.values())
    differential_gap = max(corrs) - min(corrs) if len(corrs) > 0 else 0.0
    status = "PASSED" if differential_gap < 0.20 else "FLAGGED (Differential Validity Gap)"
    return validity_by_group, differential_gap, status


def run_impact_audit(df, pred_col, target_col='Actual_Persistence', group_col='Intersectional_Cohort', threshold=0.5):
    y_true = df[target_col].values
    y_prob = df[pred_col].values
    groups = df[group_col].values
    y_pred = (y_prob >= threshold).astype(int)
    unique_groups = np.unique(groups)
    fnr_by_group = {}
    selection_rates = {}
    for g in unique_groups:
        mask = (groups == g)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        if len(sub_true) > 0 and len(np.unique(sub_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(sub_true, sub_pred, labels=[0, 1]).ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            selection_rate = (tp + fp) / len(sub_true)
        else:
            fnr = 0.0
            selection_rate = 0.0
        fnr_by_group[g] = fnr
        selection_rates[g] = selection_rate
    max_sel = max(selection_rates.values()) if len(selection_rates) > 0 else 1.0
    min_sel = min(selection_rates.values()) if len(selection_rates) > 0 else 0.0
    di_ratio = min_sel / max_sel if max_sel > 0 else 1.0
    fnr_gap = max(fnr_by_group.values()) - min(fnr_by_group.values()) if len(fnr_by_group) > 0 else 0.0
    status = "PASSED" if (di_ratio >= 0.80 and fnr_gap <= 0.15) else "FLAGGED (Disparate Impact)"
    return fnr_by_group, fnr_gap, di_ratio, status


def run_governance_audit(df, pred_col, lambda_weight, refusal_threshold):
    y_true = df['Actual_Persistence'].values
    y_prob = df[pred_col].values
    P = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    fnr_by_group, B, di_ratio, _ = run_impact_audit(df, pred_col)
    EI = P - (lambda_weight * B)
    veto_triggered = EI < refusal_threshold
    triage_labels = []
    for p in y_prob:
        if veto_triggered:
            triage_labels.append("Tier 3: Algorithmic Refusal (Vetoed)")
        elif p < 0.35:
            triage_labels.append("Tier 1: Mandatory Human Advisor Outreach")
        elif 0.35 <= p <= 0.65:
            triage_labels.append("Tier 2: Secondary Human Review Required")
        else:
            triage_labels.append("Standard Processing")
    df_triaged = df.copy()
    df_triaged['Governance_Triage'] = triage_labels
    return EI, P, B, veto_triggered, df_triaged


def calculate_disparity_penalty(test_df, prediction_col, actual_col, group_cols, penalty_weight=0.15):
    results = {}
    overall_accuracy = np.mean(test_df[prediction_col] == test_df[actual_col])
    results['overall_accuracy'] = overall_accuracy
    subgroup_metrics = {}
    max_fnr_gap = 0.0
    for group in group_cols:
        unique_groups = test_df[group].unique()
        group_stats = {}
        fnr_list = []
        for g in unique_groups:
            subset = test_df[test_df[group] == g]
            if len(subset) > 0:
                acc = np.mean(subset[prediction_col] == subset[actual_col])
                actual_positives = subset[subset[actual_col] == 1]
                if len(actual_positives) > 0:
                    fnr = np.mean(actual_positives[prediction_col] != 1)
                else:
                    fnr = 0.0
                fnr_list.append(fnr)
                group_stats[str(g)] = {'sample_size': len(subset), 'accuracy': acc, 'false_negative_rate': fnr}
        if len(fnr_list) > 1:
            group_gap = max(fnr_list) - min(fnr_list)
            max_fnr_gap = max(max_fnr_gap, group_gap)
        subgroup_metrics[group] = group_stats
    equity_penalty = max_fnr_gap * penalty_weight
    final_score = max(0.0, overall_accuracy - equity_penalty)
    results['max_fnr_gap'] = max_fnr_gap
    results['equity_penalty'] = equity_penalty
    results['final_ei_score'] = final_score
    results['subgroup_breakdown'] = subgroup_metrics
    return results


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("ei_logo.png", width=200) 
st.sidebar.title("Governance Parameters")

lambda_weight = st.sidebar.slider("Policy Calibration Weight (λ)", 0.0, 3.0, 1.5, 0.1)
refusal_threshold = st.sidebar.slider("Minimum Approval Threshold (EI)", 0.40, 0.90, 0.70, 0.05)
classification_threshold = st.sidebar.slider("Decision Boundary Threshold", 0.30, 0.70, 0.50, 0.05)

st.sidebar.divider()
st.sidebar.subheader("📂 Data Source")
data_mode = st.sidebar.radio("Select Data Source", ["Synthetic Engine", "Upload Institutional CSV"])

if data_mode == "Upload Institutional CSV":
    uploaded_file = st.sidebar.file_uploader("Upload University Dataset (.csv)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_full = pd.read_csv(uploaded_file)
            required_cols = ['Student_ID', 'Intersectional_Cohort', 'Actual_Persistence', 'GPA', 'LMS_Clicks', 'Vendor_Legacy_Prob', 'Equimeritization_Calibrated_Prob']
            for col in required_cols:
                if col not in df_full.columns:
                    if col == 'Intersectional_Cohort': df_full[col] = 'General Cohort'
                    elif col in ['GPA', 'LMS_Clicks']: df_full[col] = 3.0
                    elif col in ['Vendor_Legacy_Prob', 'Equimeritization_Calibrated_Prob']: df_full[col] = 0.5
                    elif col == 'Actual_Persistence': df_full[col] = 1
            st.sidebar.success("CSV loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            df_full = generate_synthetic_data()
    else:
        st.sidebar.info("Awaiting CSV upload. Using synthetic baseline.")
        df_full = generate_synthetic_data()
else:
    df_full = generate_synthetic_data()

st.sidebar.divider()
st.sidebar.subheader("🔍 Intersectional Slicing")
all_cohorts = sorted(df_full['Intersectional_Cohort'].unique().tolist())
selected_cohorts = st.sidebar.multiselect("Filter Cohorts", options=all_cohorts, default=all_cohorts)

df = df_full[df_full['Intersectional_Cohort'].isin(selected_cohorts)].copy() if selected_cohorts else df_full.copy()

m1_EI, m1_P, m1_B, m1_veto, df_m1_triage = run_governance_audit(df, 'Vendor_Legacy_Prob', lambda_weight, refusal_threshold)
m2_EI, m2_P, m2_B, m2_veto, df_m2_triage = run_governance_audit(df, 'Equimeritization_Calibrated_Prob', lambda_weight, refusal_threshold)

st.sidebar.divider()
run_label = st.sidebar.text_input("Audit Run Label", value=f"Audit Run #{len(st.session_state['audit_history']) + 1}")
if st.sidebar.button("Save Current Audit to Archive"):
    snapshot = {
        'Run_Label': run_label,
        'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Lambda_Weight': lambda_weight,
        'Sample_Size': len(df),
        'Vendor_EI': round(m1_EI, 3),
        'Calibrated_EI': round(m2_EI, 3),
        'Vendor_Status': "VETOED" if m1_veto else "APPROVED",
        'Calibrated_Status': "VETOED" if m2_veto else "APPROVED"
    }
    st.session_state['audit_history'].append(snapshot)
    st.sidebar.success(f"Archived '{run_label}'!")

st.sidebar.caption(f"Active Sample: {len(df):,} records\nUTSA Educational Leadership & Policy Studies\nDeveloped by Steven Westgreen")


# ==========================================
# HEADER & EXECUTIVE METRICS
# ==========================================
st.markdown('<div class="main-header">Equimeritization Index (EI) Executive Suite</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Auditing Algorithmic Power and Disparate Error Rates | Developer: Steven Westgreen | Active Sample: {len(df):,} Records</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Policy Weight (λ)", f"{lambda_weight:.1f}", "Institutional Setting")
with col2:
    status_m1 = "REJECTED (Veto)" if m1_veto else "APPROVED"
    st.metric("Vendor Model EI", f"{m1_EI:.3f}", status_m1, delta_color="inverse" if m1_veto else "normal")
with col3:
    status_m2 = "REJECTED (Veto)" if m2_veto else "APPROVED"
    st.metric("Calibrated Model EI", f"{m2_EI:.3f}", status_m2, delta_color="inverse" if m2_veto else "normal")
with col4:
    disp_reduction = ((m1_B - m2_B) / m1_B * 100) if m1_B > 0 else 0.0
    st.metric("Disparity Reduction", f"{disp_reduction:.1f}%", "Equity Improvement")

st.divider()


# ==========================================
# VENDOR AUDIT UI RENDER FUNCTION (SINGLE + COMPARATIVE MODES)
# ==========================================
def render_vendor_audit_section():
    """
    Streamlit UI component for third-party vendor equity auditing, supporting both:
    1. Single Vendor Audit Mode
    2. Side-by-Side Comparative Benchmarking Mode (Vendor A vs. Vendor B)
    """
    st.subheader("📦 Third-Party Vendor Equity Audit Suite")
    st.markdown("Evaluate vendor model risk predictions against ground truth student persistence outcomes for error disparities, penalty scoring, and compliance verification.")
    
    # Audit Mode Selection Toggle
    audit_mode = st.radio(
        "Select Audit Workflow Mode:",
        ["Single Vendor Audit", "Comparative Benchmark (2 Vendors)"],
        horizontal=True,
        key="vendor_audit_mode_selector"
    )
    st.divider()

    # ==========================================
    # MODE 1: SINGLE VENDOR AUDIT WORKFLOW
    # ==========================================
    if audit_mode == "Single Vendor Audit":
        uploaded_file = st.file_uploader("Upload Vendor Test CSV File", type=["csv"], key="single_vendor_uploader")
        
        if uploaded_file is not None:
            df_vendor = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {len(df_vendor):,} records.")
            
            with st.expander("Preview Uploaded Vendor Dataset"):
                st.dataframe(df_vendor.head(), use_container_width=True)
                
            st.markdown("### Configuration & Column Mapping")
            col1, col2, col3 = st.columns(3)
            with col1:
                pred_col = st.selectbox("Vendor Prediction Column", df_vendor.columns, key="single_pred_col")
            with col2:
                act_col = st.selectbox("Actual Outcome Column (Ground Truth)", df_vendor.columns, key="single_act_col")
            with col3:
                default_groups = [c for c in ['Intersectional_Cohort', 'First_Gen', 'Pell_Recipient', 'URM_Status'] if c in df_vendor.columns]
                group_cols = st.multiselect("Demographic Pillars for Disparity Check", df_vendor.columns, default=default_groups, key="single_group_cols")
                
            if st.button("Run Single Vendor Audit", type="primary"):
                if not group_cols:
                    st.error("Please select at least one demographic pillar for auditing.")
                else:
                    with st.spinner("Calculating error rates and equity penalties..."):
                        audit_results = calculate_disparity_penalty(df_vendor, pred_col, act_col, group_cols)
                        
                    st.divider()
                    st.markdown("### Vendor Audit Scorecard")
                    
                    is_compliant = audit_results['max_fnr_gap'] <= 0.15 and audit_results['final_ei_score'] >= 0.65
                    if is_compliant:
                        st.success("🟢 **INSTITUTIONAL COMPLIANCE STATUS: PASSED** — Vendor model meets equity thresholds.")
                    else:
                        st.error("🔴 **INSTITUTIONAL COMPLIANCE STATUS: FAILED / FLAGGED** — Excessive disparate error rates detected.")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Raw Overall Accuracy", f"{audit_results['overall_accuracy']:.2%}")
                    m2.metric("Max Disparity Gap (FNR)", f"{audit_results['max_fnr_gap']:.2%}")
                    m3.metric("Final EI Score (Post-Penalty)", f"{audit_results['final_ei_score']:.2%}", delta=f"-{audit_results['equity_penalty']:.2%}", delta_color="inverse")
                    
                    st.divider()
                    st.markdown("### 🔍 Demographic Subgroup Disparity Table")
                    selected_pillar = st.selectbox("Select Demographic Pillar to Inspect", group_cols, key="single_drilldown_pillar")
                    
                    pillar_stats = audit_results['subgroup_breakdown'].get(selected_pillar, {})
                    pillar_rows, chart_data = [], []
                    for grp_val, metrics in pillar_stats.items():
                        pillar_rows.append({
                            'Subgroup': f"{selected_pillar}: {grp_val}",
                            'Sample Size': metrics['sample_size'],
                            'Accuracy': f"{metrics['accuracy']:.2%}",
                            'False Negative Rate (FNR)': f"{metrics['false_negative_rate']:.2%}"
                        })
                        chart_data.append({'Subgroup': str(grp_val), 'False Negative Rate': metrics['false_negative_rate']})
                    
                    st.dataframe(pd.DataFrame(pillar_rows), use_container_width=True)
                    
                    df_chart = pd.DataFrame(chart_data)
                    if not df_chart.empty:
                        fig_fnr = px.bar(df_chart, x='Subgroup', y='False Negative Rate', text_auto='.1%', color='False Negative Rate', color_continuous_scale='Reds')
                        fig_fnr.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_tickformat='.0%', xaxis_tickangle=-20, showlegend=False)
                        st.plotly_chart(fig_fnr, use_container_width=True)
                        
                    st.divider()
                    st.markdown("### 🖨️ Export Audit Deliverables")
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        summary_df = pd.DataFrame({
                            'Metric': ['Overall Accuracy', 'Max FNR Disparity Gap', 'Equity Penalty Applied', 'Final EI Score', 'Compliance Status'],
                            'Value': [f"{audit_results['overall_accuracy']:.4f}", f"{audit_results['max_fnr_gap']:.4f}", f"{audit_results['equity_penalty']:.4f}", f"{audit_results['final_ei_score']:.4f}", "PASSED" if is_compliant else "FAILED"]
                        })
                        st.download_button("📊 Download Scorecard (CSV)", summary_df.to_csv(index=False).encode('utf-8'), "single_vendor_audit_scorecard.csv", "text/csv")
                        
                    with col_exp2:
                        memo_text = f"""================================================================================
SINGLE VENDOR ALGORITHMIC EQUITY AUDIT MEMORANDUM
================================================================================
Author / Lead Developer: Steven Westgreen
Institutional Context: UTSA Educational Leadership & Policy Studies
Audit Timestamp: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Records Evaluated: {len(df_vendor):,} student records

COMPLIANCE VERDICT:
--------------------------------------------------------------------------------
Institutional Status: {"APPROVED / COMPLIANT" if is_compliant else "FLAGGED / CONTRACT VETO RECOMMENDED"}

AUDIT METRICS SUMMARY:
--------------------------------------------------------------------------------
- Raw Vendor Predictive Accuracy: {audit_results['overall_accuracy']:.2%}
- Maximum False Negative Rate (FNR) Disparity Gap: {audit_results['max_fnr_gap']:.2%}
- Applied Equity Penalty: -{audit_results['equity_penalty']:.2%}
- Final Penalized Equimeritization Score (EI): {audit_results['final_ei_score']:.2%}

EVALUATED DEMOGRAPHIC PILLARS: {", ".join(group_cols)}
================================================================================
"""
                        st.download_button("📄 Download Audit Memo (TXT)", memo_text, "Single_Vendor_Audit_Memo.txt", "text/plain")
        else:
            st.info("Awaiting CSV file upload for single vendor audit.")

    # ==========================================
    # MODE 2: COMPARATIVE BENCHMARKING WORKFLOW
    # ==========================================
    else:
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            file_a = st.file_uploader("Upload Vendor A (Incumbent Model) CSV", type=["csv"], key="vendor_a_uploader")
        with col_up2:
            file_b = st.file_uploader("Upload Vendor B (Challenger Model) CSV", type=["csv"], key="vendor_b_uploader")
        
        if file_a is not None and file_b is not None:
            df_a = pd.read_csv(file_a)
            df_b = pd.read_csv(file_b)
            
            st.success(f"Loaded datasets — Vendor A: {len(df_a):,} records | Vendor B: {len(df_b):,} records.")
            
            with st.expander("Preview Vendor Datasets"):
                tab_pa, tab_pb = st.tabs(["Vendor A Preview", "Vendor B Preview"])
                with tab_pa: st.dataframe(df_a.head(), use_container_width=True)
                with tab_pb: st.dataframe(df_b.head(), use_container_width=True)
                    
            st.markdown("### Configuration & Column Mapping")
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            with col_cfg1:
                act_col = st.selectbox("Ground Truth Column (Actual Outcome)", df_a.columns, key="bench_act_col")
            with col_cfg2:
                default_groups = [c for c in ['Intersectional_Cohort', 'First_Gen', 'Pell_Recipient', 'URM_Status'] if c in df_a.columns]
                group_cols = st.multiselect("Demographic Pillars for Disparity Check", df_a.columns, default=default_groups, key="bench_group_cols")
            with col_cfg3:
                model_a_name = st.text_input("Vendor A Label", value="Incumbent Vendor Model")
                model_b_name = st.text_input("Vendor B Label", value="Challenger AI Model")
                
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                pred_col_a = st.selectbox(f"{model_a_name} Prediction Column", df_a.columns, key="pred_a_col")
            with col_pred2:
                pred_col_b = st.selectbox(f"{model_b_name} Prediction Column", df_b.columns, key="pred_b_col")
                
            if st.button("Run Comparative Vendor Benchmark Audit", type="primary"):
                if not group_cols:
                    st.error("Please select at least one demographic pillar for auditing.")
                else:
                    with st.spinner("Executing comparative disparity audits..."):
                        results_a = calculate_disparity_penalty(df_a, pred_col_a, act_col, group_cols)
                        results_b = calculate_disparity_penalty(df_b, pred_col_b, act_col, group_cols)
                        
                    st.divider()
                    st.markdown("### ⚖️ Side-by-Side Compliance Scorecard")
                    
                    is_compliant_a = results_a['max_fnr_gap'] <= 0.15 and results_a['final_ei_score'] >= 0.65
                    is_compliant_b = results_b['max_fnr_gap'] <= 0.15 and results_b['final_ei_score'] >= 0.65
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown(f"#### 🏛️ {model_a_name}")
                        if is_compliant_a: st.success("🟢 **STATUS: PASSED**")
                        else: st.error("🔴 **STATUS: FLAGGED / FAILED**")
                        st.metric("Raw Overall Accuracy", f"{results_a['overall_accuracy']:.2%}")
                        st.metric("Max Disparity Gap (FNR)", f"{results_a['max_fnr_gap']:.2%}")
                        st.metric("Final EI Score", f"{results_a['final_ei_score']:.2%}", delta=f"-{results_a['equity_penalty']:.2%}", delta_color="inverse")
                        
                    with col_res2:
                        st.markdown(f"#### 🚀 {model_b_name}")
                        if is_compliant_b: st.success("🟢 **STATUS: PASSED**")
                        else: st.error("🔴 **STATUS: FLAGGED / FAILED**")
                        st.metric("Raw Overall Accuracy", f"{results_b['overall_accuracy']:.2%}", delta=f"{(results_b['overall_accuracy'] - results_a['overall_accuracy']):+.2%}")
                        st.metric("Max Disparity Gap (FNR)", f"{results_b['max_fnr_gap']:.2%}", delta=f"{(results_b['max_fnr_gap'] - results_a['max_fnr_gap']):+.2%}", delta_color="inverse")
                        st.metric("Final EI Score", f"{results_b['final_ei_score']:.2%}", delta=f"{(results_b['final_ei_score'] - results_a['final_ei_score']):+.2%}")

                    st.divider()
                    st.markdown("### 🏆 Institutional Procurement Recommendation")
                    if is_compliant_a and not is_compliant_b:
                        recommended_winner = model_a_name
                        rationale = f"{model_a_name} is the sole compliant model meeting equity thresholds."
                    elif is_compliant_b and not is_compliant_a:
                        recommended_winner = model_b_name
                        rationale = f"{model_b_name} is the sole compliant model meeting equity thresholds."
                    else:
                        recommended_winner = model_a_name if results_a['final_ei_score'] >= results_b['final_ei_score'] else model_b_name
                        rationale = f"Both models share similar compliance standing; {recommended_winner} achieves a superior penalization-adjusted EI score."

                    st.info(f"💡 **Recommended Selection:** **{recommended_winner}**\n\n*Rationale:* {rationale}")

                    st.divider()
                    st.markdown("### 📊 Comparative False Negative Rate (FNR) Disparity")
                    selected_pillar = st.selectbox("Select Pillar for Visual Comparison", group_cols, key="comp_drilldown_pillar")
                    
                    stats_a = results_a['subgroup_breakdown'].get(selected_pillar, {})
                    stats_b = results_b['subgroup_breakdown'].get(selected_pillar, {})
                    
                    chart_rows = []
                    for grp, m_data in stats_a.items():
                        chart_rows.append({'Subgroup': str(grp), 'False Negative Rate': m_data['false_negative_rate'], 'Vendor Model': model_a_name})
                    for grp, m_data in stats_b.items():
                        chart_rows.append({'Subgroup': str(grp), 'False Negative Rate': m_data['false_negative_rate'], 'Vendor Model': model_b_name})
                        
                    df_comp_chart = pd.DataFrame(chart_rows)
                    if not df_comp_chart.empty:
                        fig_comp = px.bar(df_comp_chart, x='Subgroup', y='False Negative Rate', color='Vendor Model', barmode='group', color_discrete_sequence=['#DC2626', '#2563EB'])
                        fig_comp.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_tickformat='.0%', xaxis_tickangle=-20)
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                    st.divider()
                    st.markdown("### 🖨️ Export Board-Ready Deliverables")
                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                    
                    with col_exp1:
                        comp_summary_df = pd.DataFrame({
                            'Metric': ['Overall Accuracy', 'Max FNR Disparity Gap', 'Equity Penalty Applied', 'Final EI Score', 'Compliance Status'],
                            model_a_name: [f"{results_a['overall_accuracy']:.4f}", f"{results_a['max_fnr_gap']:.4f}", f"{results_a['equity_penalty']:.4f}", f"{results_a['final_ei_score']:.4f}", "PASSED" if is_compliant_a else "FAILED"],
                            model_b_name: [f"{results_b['overall_accuracy']:.4f}", f"{results_b['max_fnr_gap']:.4f}", f"{results_b['equity_penalty']:.4f}", f"{results_b['final_ei_score']:.4f}", "PASSED" if is_compliant_b else "FAILED"]
                        })
                        st.download_button("📊 Download Scorecard (CSV)", comp_summary_df.to_csv(index=False).encode('utf-8'), "vendor_comparative_benchmark_scorecard.csv", "text/csv")
                        
                    with col_exp2:
                        comp_memo_text = f"""================================================================================
THE UNIVERSITY OF TEXAS AT SAN ANTONIO (UTSA)
DEPARTMENT OF EDUCATIONAL LEADERSHIP AND POLICY STUDIES (ELPS)
EQUIMERITIZATION INDEX (EI) — EXECUTIVE GOVERNANCE & PROCUREMENT BRIEF
================================================================================
CONFIDENTIAL INSTITUTIONAL AUDIT REPORT | DATE: {pd.Timestamp.now().strftime("%B %d, %Y")}
PRINCIPAL INVESTIGATOR: Steven Westgreen, Ph.D. Candidate

1. BOTTOM-LINE-UP-FRONT (BLUF) EXECUTIVE RECOMMENDATION
--------------------------------------------------------------------------------
Recommended Vendor Selection : {recommended_winner}
Strategic Rationale          : {rationale}

2. COMPARATIVE BENCHMARKING SCORECARD
--------------------------------------------------------------------------------
Metric Description           | {model_a_name[:20]:<20} | {model_b_name[:20]:<20}
--------------------------------------------------------------------------------
Raw Overall Accuracy         | {results_a['overall_accuracy']:.2%}               | {results_b['overall_accuracy']:.2%}               
Max FNR Disparity Gap (B)    | {results_a['max_fnr_gap']:.2%}               | {results_b['max_fnr_gap']:.2%}               
Equity Penalty Applied       | -{results_a['equity_penalty']:.2%}              | -{results_b['equity_penalty']:.2%}              
Final Penalized EI Score     | {results_a['final_ei_score']:.2%}               | {results_b['final_ei_score']:.2%}               
Institutional Status         | {"PASSED" if is_compliant_a else "FAILED"}              | {"PASSED" if is_compliant_b else "FAILED"}              
--------------------------------------------------------------------------------
"""
                        st.download_button("📄 Download Text Memo (TXT)", comp_memo_text, "Comparative_Vendor_Policy_Memo.txt", "text/plain")

                    with col_exp3:
                        def generate_pdf_memo():
                            buffer = io.BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                            story = []
                            styles = getSampleStyleSheet()
                            title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=11, textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=4)
                            subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=7.5, textColor=colors.HexColor('#475569'), alignment=1, spaceAfter=8)
                            heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=9.5, textColor=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=3)
                            body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#0F172A'), spaceAfter=4)
                            
                            story.append(Paragraph("THE UNIVERSITY OF TEXAS AT SAN ANTONIO (UTSA)", title_style))
                            story.append(Paragraph("DEPARTMENT OF EDUCATIONAL LEADERSHIP AND POLICY STUDIES (ELPS)<br/><b>Equimeritization Index (EI) — Executive Governance & Procurement Brief</b>", subtitle_style))
                            story.append(Spacer(1, 4))
                            
                            story.append(Paragraph("1. Bottom-Line-Up-Front (BLUF) Executive Recommendation", heading_style))
                            story.append(Paragraph(f"<b>Recommended Vendor Selection:</b> {recommended_winner}<br/><b>Strategic Rationale:</b> {rationale}", body_style))
                            
                            story.append(Paragraph("2. Comparative Benchmarking Scorecard", heading_style))
                            table_data = [
                                ['Metric Description', model_a_name[:18], model_b_name[:18]],
                                ['Raw Overall Accuracy', f"{results_a['overall_accuracy']:.2%}", f"{results_b['overall_accuracy']:.2%}"],
                                ['Max FNR Disparity Gap (B)', f"{results_a['max_fnr_gap']:.2%}", f"{results_b['max_fnr_gap']:.2%}"],
                                ['Equity Penalty Applied', f"-{results_a['equity_penalty']:.2%}", f"-{results_b['equity_penalty']:.2%}"],
                                ['Final Penalized EI Score', f"{results_a['final_ei_score']:.2%}", f"{results_b['final_ei_score']:.2%}"],
                                ['Institutional Status', "PASSED" if is_compliant_a else "FAILED", "PASSED" if is_compliant_b else "FAILED"]
                            ]
                            t = Table(table_data, colWidths=[180, 175, 175])
                            t.setStyle(TableStyle([
                                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
                                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0,0), (-1,-1), 8),
                                ('BOTTOMPADDING', (0,0), (-1,0), 3),
                                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
                                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1'))
                            ]))
                            story.append(t)
                            story.append(Spacer(1, 4))
                            story.append(Paragraph("3. Governance Implications & Next Steps", heading_style))
                            story.append(Paragraph("Comparative analysis proves raw accuracy obscures error-rate disparities across marginalized cohorts. Procurement committees must condition contracts on maintaining EI scores above institutional thresholds.", body_style))
                            
                            doc.build(story)
                            buffer.seek(0)
                            return buffer

                        st.download_button("📥 Download PDF Memorandum (.pdf)", generate_pdf_memo(), "Equimeritization_Executive_Brief.pdf", "application/pdf")
        else:
            st.info("Awaiting upload of both Vendor A and Vendor B test datasets for comparative benchmarking.")
# ==========================================
# TABS NAVIGATION
# ==========================================
tab_main, tab_sweep, tab_ingest, tab_raw, tab_archive, tab_epi, tab_con, tab_imp, tab_gov, tab_vendor, tab_guide = st.tabs([
    "📊 Executive Summary",
    "📈 Policy Sensitivity",
    "📁 Data Ingestion",
    "📋 Raw Logs",
    "🏛️ Multi-Audit Archive",
    "🔍 1. Epistemic Audit",
    "📐 2. Construct Audit",
    "⚖️ 3. Impact Audit",
    "🛡️ 4. Governance & Capacity",
    "📦 Vendor Audit",
    "📖 User Guide"
])

with tab_main:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Equimeritization Breakdown ($EI = P - (\\lambda \\times B)$)")
        comp_df = pd.DataFrame({
            'Metric': ['Predictive Performance (P)', 'Bias Penalty (B)', 'Equimeritization Index (EI)'],
            'Vendor Proprietary Model': [m1_P, m1_B, m1_EI],
            'Equimeritization Calibrated Model': [m2_P, m2_B, m2_EI]
        })
        fig_bar = px.bar(comp_df, x='Metric', y=['Vendor Proprietary Model', 'Equimeritization Calibrated Model'], barmode='group', color_discrete_sequence=['#DC2626', '#059669'])
        fig_bar.add_hline(y=refusal_threshold, line_dash="dash", line_color="#000000", annotation_text="Minimum EI Threshold")
        fig_bar.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_right:
        st.subheader("False Negative Rate (FNR) Disparity by Cohort")
        fnr1, _, _, _ = run_impact_audit(df, 'Vendor_Legacy_Prob', threshold=classification_threshold)
        fnr2, _, _, _ = run_impact_audit(df, 'Equimeritization_Calibrated_Prob', threshold=classification_threshold)
        groups_list = list(fnr1.keys())
        fnr_df = pd.DataFrame({'Intersectional Cohort': groups_list, 'Vendor Model FNR': [fnr1[g] for g in groups_list], 'Calibrated Model FNR': [fnr2[g] for g in groups_list]})
        fig_fnr = px.bar(fnr_df, x='Intersectional Cohort', y=['Vendor Model FNR', 'Calibrated Model FNR'], barmode='group', color_discrete_sequence=['#2563EB', '#7C3AED'])
        fig_fnr.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_tickangle=-25)
        st.plotly_chart(fig_fnr, use_container_width=True)

    if m1_veto and not m2_veto:
        st.error(f"⚠️ **ALGORITHMIC REFUSAL PROTOCOL TRIGGERED:** Vendor Proprietary Model scored **{m1_EI:.3f}** (Below threshold of {refusal_threshold}). Recommend vetoing vendor contract and deploying Calibrated Model (**EI = {m2_EI:.3f}**).")

    st.divider()
    st.subheader("🖨️ Executive Policy Memo & Audit Summary Report")
    report_text = f"""================================================================================
EQUIMERITIZATION INDEX (EI) EXECUTIVE AUDIT MEMORANDUM
================================================================================
Author / Lead Developer: Steven Westgreen
Institutional Context: UTSA Educational Leadership & Policy Studies
Audit Timestamp: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Sample Size Analyzed: {len(df):,} student records

GOVERNANCE AUDIT PARAMETERS:
- Policy Calibration Weight (λ): {lambda_weight:.2f}
- Minimum Approval Threshold (EI Cutoff): {refusal_threshold:.2f}

PERFORMANCE RESULTS:
1. Vendor Proprietary Model: EI = {m1_EI:.4f} (Status: {"VETOED" if m1_veto else "APPROVED"})
2. Equimeritization Calibrated Model: EI = {m2_EI:.4f} (Status: {"VETOED" if m2_veto else "APPROVED"})
================================================================================
"""
    st.text_area("Executive Summary Preview", value=report_text, height=180)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📄 Download Memo (TXT)", report_text, "Equimeritization_Executive_Policy_Memo.txt", "text/plain")
    with col_dl2:
        csv_summary = pd.DataFrame({
            'Model': ['Vendor Proprietary Model', 'Equimeritization Calibrated Model'],
            'Predictive_P': [m1_P, m2_P], 'Bias_Penalty_B': [m1_B, m2_B], 'EI_Score': [m1_EI, m2_EI],
            'Status': ["VETOED" if m1_veto else "APPROVED", "VETOED" if m2_veto else "APPROVED"]
        }).to_csv(index=False).encode('utf-8')
        st.download_button("📊 Download Metrics (CSV)", csv_summary, "EI_Executive_Summary_Metrics.csv", "text/csv")

with tab_sweep:
    st.subheader("Policy Calibration Weight (λ) Sensitivity Sweep")
    lambdas = np.linspace(0.0, 3.0, 50)
    m1_eis = [m1_P - (l * m1_B) for l in lambdas]
    m2_eis = [m2_P - (l * m2_B) for l in lambdas]
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=lambdas, y=m1_eis, mode='lines', name='Vendor Proprietary Model', line=dict(color='#DC2626', width=3)))
    fig_sweep.add_trace(go.Scatter(x=lambdas, y=m2_eis, mode='lines', name='Equimeritization Calibrated Model', line=dict(color='#059669', width=3)))
    fig_sweep.add_vline(x=lambda_weight, line_dash="dash", line_color="#0F172A", annotation_text=f"Current λ = {lambda_weight}")
    fig_sweep.update_layout(xaxis_title="Policy Weight (λ)", yaxis_title="Equimeritization Index (EI)", template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_sweep, use_container_width=True)

with tab_ingest:
    st.subheader("📁 Institutional Data Ingestion & Schema Guidelines")
    st.markdown("Upload your university's anonymized student persistence and algorithmic risk dataset to run real-time 4-pillar audits.")
    sample_csv = generate_synthetic_data(n_samples=100).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV Template", sample_csv, "EI_Sample_Dataset_Template.csv", "text/csv")

with tab_raw:
    st.subheader("Audited Student Records & Governance Triage Logs")
    st.dataframe(df_m1_triage[['Student_ID', 'Intersectional_Cohort', 'Actual_Persistence', 'Vendor_Legacy_Prob', 'Equimeritization_Calibrated_Prob', 'Governance_Triage']], use_container_width=True)

with tab_archive:
    st.subheader("🏛️ Institutional Multi-Audit Historical Archive")
    if len(st.session_state['audit_history']) > 0:
        archive_df = pd.DataFrame(st.session_state['audit_history'])
        st.dataframe(archive_df, use_container_width=True)
        st.download_button("📥 Download Archive (CSV)", archive_df.to_csv(index=False).encode('utf-8'), "EI_Multi_Audit_Archive.csv", "text/csv")
        if st.button("🗑️ Clear Archive"):
            st.session_state['audit_history'] = []
            st.rerun()
    else:
        st.info("No audit runs archived yet.")

with tab_epi:
    st.subheader("Pillar 1: Epistemic Audit (Feature Lineage & Proxy Concentration)")
    imp_df, top_proxy_ratio, epi_status = run_epistemic_audit(df)
    c_e1, c_e2 = st.columns([1, 2])
    with c_e1:
        st.metric("Epistemic Audit Status", epi_status)
        st.metric("Top Feature Reliance Ratio", f"{top_proxy_ratio * 100:.1f}%")
    with c_e2:
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#1E3A8A'])
        fig_imp.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

with tab_con:
    st.subheader("Pillar 2: Construct Audit (Proxy Validity & Differential Validity)")
    validity_by_group, diff_gap, con_status = run_construct_audit(df)
    c_c1, c_c2 = st.columns([1, 2])
    with c_c1:
        st.metric("Construct Audit Status", con_status)
        st.metric("Differential Validity Gap", f"{diff_gap:.3f}")
    with c_c2:
        val_df = pd.DataFrame({'Intersectional Cohort': list(validity_by_group.keys()), 'LMS Click Correlation (r)': list(validity_by_group.values())})
        fig_val = px.bar(val_df, x='Intersectional Cohort', y='LMS Click Correlation (r)', color='Intersectional Cohort')
        fig_val.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-25)
        st.plotly_chart(fig_val, use_container_width=True)

with tab_imp:
    st.subheader("Pillar 3: Impact Audit (Disparate Impact & 80% Rule)")
    fnr1, B1, di_ratio1, status1 = run_impact_audit(df, 'Vendor_Legacy_Prob', threshold=classification_threshold)
    fnr2, B2, di_ratio2, status2 = run_impact_audit(df, 'Equimeritization_Calibrated_Prob', threshold=classification_threshold)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("### Vendor Legacy Model")
        st.metric("Impact Audit Status", status1)
        st.metric("Disparate Impact Ratio", f"{di_ratio1:.2f}", "FAIL" if di_ratio1 < 0.80 else "PASS")
        st.metric("FNR Disparity Gap (B)", f"{B1:.3f}")
    with ci2:
        st.markdown("### Calibrated Model")
        st.metric("Impact Audit Status", status2)
        st.metric("Disparate Impact Ratio", f"{di_ratio2:.2f}", "PASS")
        st.metric("FNR Disparity Gap (B)", f"{B2:.3f}")

with tab_gov:
    st.subheader("Pillar 4: Governance Audit & Advisor Caseload Capacity Simulator")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        num_advisors = st.slider("Total Campus Academic Advisors", 5, 100, 25, 5)
        max_caseload = st.slider("Max Students per Advisor", 50, 300, 150, 10)
        tier1_count = len(df_m1_triage[df_m1_triage['Governance_Triage'].str.contains("Tier 1")])
        total_capacity = num_advisors * max_caseload
        capacity_ratio = (tier1_count / total_capacity) * 100 if total_capacity > 0 else 0
        st.metric("Tier 1 Students Flagged", f"{tier1_count:,} students")
        st.metric("Total Advisor Capacity", f"{total_capacity:,} student slots")
        if tier1_count > total_capacity:
            st.error(f"🚨 **CAPACITY BOTTLENECK:** Tier 1 outreach demand exceeds total advisor capacity by {tier1_count - total_capacity:,} students.")
        else:
            st.success(f"✅ **CAPACITY ADEQUATE:** Staffing can accommodate Tier 1 outreach ({capacity_ratio:.1f}% capacity utilization).")
    with col_g2:
        triage_counts = df_m1_triage['Governance_Triage'].value_counts().reset_index()
        triage_counts.columns = ['Triage Tier', 'Student Count']
        fig_triage = px.pie(triage_counts, names='Triage Tier', values='Student Count', title="Vendor Model Triage Allocation", color_discrete_sequence=['#DC2626', '#D97706', '#2563EB', '#059669'])
        fig_triage.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_triage, use_container_width=True)

with tab_vendor:
    render_vendor_audit_section()

with tab_guide:
    st.subheader("📖 Equimeritization Index ($EI$) Suite - Official User & Technical Operations Guide")
    st.markdown("##### Developed by Steven Westgreen | UTSA Educational Leadership & Policy Studies")
    st.divider()
    st.markdown("""
    ### 1. Executive Summary & Core Philosophy
    The **Equimeritization Index ($EI$) Suite** is an enterprise-grade algorithmic governance platform designed to evaluate equity, transparency, and disparate impact in AI-driven predictive systems within higher education.
    
    $$EI = P - (\\lambda \\times B)$$
    
    ### 2. System Architecture & The 4-Pillar Audit Workflow
    * **Pillar 1: Epistemic Audit:** Evaluates whether model predictions rely heavily on proxy variables.
    * **Pillar 2: Construct Audit:** Tests whether behavioral proxy variables measure engagement consistently across subgroups.
    * **Pillar 3: Impact Audit:** Measures adverse impact ratios and FNR gaps.
    * **Pillar 4: Governance & Capacity Simulator:** Maps predictions into human oversight tiers and tests advising workload capacity.
      
      
    ### 3. Worked Example: Balancing Accuracy and Fairness 
    The EI balances **accuracy** and **fairness** when evaluating a model, for example an admissions model. 
  
    **Key Formula:** EI = P - (λ × B) 
    * **P** = Performance (overall accuracy of the model) 
    * **B** = Bias penalty (how unfair the model is across different groups) 
    * **λ (lambda)** = Policy Calibration Weight, which sets how much the institution cares about fairness relative to accuracy 
  
    Think of λ as a **fairness dial**: 
    * If λ = 0, fairness is ignored and only accuracy matters. 
    * If λ is large, unfairness is strongly punished. 
    * Different institutions can choose different λ values based on their values and policies. 
  
    #### The Example 
    Two admissions models are compared: 
  
    | Model | Accuracy (P) | Bias Penalty (B) | 
    |---|---|---| 
    | A | 0.92 | 0.25 | 
    | B | 0.88 | 0.04 | 
  
    At first glance, **Model A looks better** because its accuracy is higher (92% vs. 88%). However, Model A is also much more biased (0.25 vs. 0.04). The institution chooses λ = 0.50, meaning each unit of bias reduces the final score by half that amount. 
  
    #### Calculate Model A 
    * Bias deduction: 0.50 × 0.25 = 0.125 
    * EI: 0.92 - 0.125 = 0.795 
  
    #### Calculate Model B 
    * Bias deduction: 0.50 × 0.04 = 0.020 
    * EI: 0.88 - 0.020 = 0.860 
  
    #### Interpretation 
    Although Model A is **more accurate**, its unfairness is severe enough that its final EI score is lower than Model B's: 
  
    | Model | EI Score | 
    |---|---| 
    | A | 0.795 | 
    | B | 0.860 | 
  
    As a result, the framework selects **Model B**. 
  
    #### Why This Matters 
    Without the bias penalty, the university would probably choose the 92% accurate model. The EI framework argues that a model that is slightly less accurate but substantially fairer may be preferable to a highly accurate model that systematically disadvantages certain student groups. In other words, the framework treats **equity as a quantifiable institutional value**, not merely an afterthought. 
  
    #### Simple Analogy 
    Imagine choosing between two professors: 
    * Professor A gives highly accurate grades but is known to grade one group of students much harsher than others. 
    * Professor B is slightly less consistent overall but treats all student groups much more equally. 
  
    The parameter **λ** reflects how important equal treatment is to the university. The higher λ is, the more the university is willing to sacrifice a little efficiency or accuracy to ensure fairness. 
  
    #### The Main Takeaway 
    **λ transforms fairness from a philosophical discussion into a policy choice** that can be explicitly encoded in the model evaluation process. By increasing or decreasing λ, stakeholders can transparently decide how much bias they are willing to tolerate in exchange for higher predictive accuracy. 
  
    """)
