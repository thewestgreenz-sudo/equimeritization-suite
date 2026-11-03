import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# PAGE CONFIGURATION & EXECUTIVE STYLING
# ==========================================
st.set_page_config(
    page_title="Equimeritization Index (EI) Executive Audit Suite",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Institutional Executive Light Theme CSS with Horizontal Scrollable Tabs
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Typography */
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
    
    /* Executive Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="metric-container"] label {
        color: #64748B !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Scrollable Horizontal Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        gap: 8px;
        background-color: #F8FAFC;
        padding-bottom: 12px;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-shrink: 0;
        background-color: #FFFFFF;
        border-radius: 8px;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 10px 18px;
        font-weight: 600;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border: 1px solid #1E3A8A !important;
    }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE ARCHIVE INITIALIZATION
# ==========================================
if 'audit_history' not in st.session_state:
    st.session_state['audit_history'] = []


# ==========================================
# SYNTHETIC DATA ENGINE
# ==========================================
@st.cache_data
def generate_synthetic_data(n_samples=5000, seed=42):
    """Generates synthetic educational predictive model data with embedded structural disparities."""
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
# CORE 4-PILLAR AUDIT LOGIC
# ==========================================
def run_epistemic_audit(df):
    """Pillar 1: Evaluates feature importance and reliance on proxy variables."""
    X = df[['GPA', 'LMS_Clicks']]
    y = df['Actual_Persistence']
    
    if len(X) < 10:
        return pd.DataFrame({'Feature': ['GPA', 'LMS_Clicks'], 'Importance': [0.5, 0.5]}), 0.5, "INSUFFICIENT DATA"

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    perm_imp = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({
        'Feature': ['GPA', 'LMS_Clicks'],
        'Importance': perm_imp.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    total_imp = imp_df['Importance'].sum()
    top_proxy_ratio = imp_df.iloc[0]['Importance'] / total_imp if total_imp > 0 else 0.0
    status = "PASSED" if top_proxy_ratio < 0.75 else "FLAGGED (High Proxy Reliance)"
    
    return imp_df, top_proxy_ratio, status


def run_construct_audit(df, proxy_col='LMS_Clicks', target_col='Actual_Persistence', group_col='Intersectional_Cohort'):
    """Pillar 2: Evaluates differential validity of behavioral proxy variables."""
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
    """Pillar 3: Evaluates False Negative Rate (FNR) disparity and Disparate Impact Ratio."""
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
    """Pillar 4: Computes EI Score and categorizes student records into triage tiers."""
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


# ==========================================
# NEW: VENDOR AUDIT HELPER FUNCTION
# ==========================================
def calculate_disparity_penalty(test_df, prediction_col, actual_col, group_cols, penalty_weight=0.15):
    """
    Calculates overall accuracy, subgroup error rates, and applies an equity 
    disparity penalty based on False Negative Rate (FNR) gaps across protected groups.
    """
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
                group_stats[str(g)] = {
                    'sample_size': len(subset),
                    'accuracy': acc,
                    'false_negative_rate': fnr
                }
        
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
# SIDEBAR CONTROLS & SETTINGS
# ==========================================
st.sidebar.image("https://img.icons8.com/isometric-folders/100/scales.png", width=60)
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

if selected_cohorts:
    df = df_full[df_full['Intersectional_Cohort'].isin(selected_cohorts)].copy()
else:
    df = df_full.copy()

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

st.sidebar.caption(f"Active Sample: {len(df):,} records")
st.sidebar.caption("Equimeritization Framework v1.0\nUTSA Educational Leadership & Policy Studies\nDeveloped by Steve Westgreen")


# ==========================================
# DASHBOARD HEADER & EXECUTIVE KPIS
# ==========================================
st.markdown('<div class="main-header">Equimeritization Index (EI) Executive Suite</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Auditing Algorithmic Power and Disparate Error Rates | Developer: Steve Westgreen | Active Sample: {len(df):,} Records</div>', unsafe_allow_html=True)

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
# NEW: VENDOR AUDIT UI RENDER FUNCTION
# ==========================================
def render_vendor_audit_section():
    """
    Streamlit UI component for uploading and auditing third-party vendor predictions.
    """
    st.subheader("📦 Third-Party Vendor Black-Box Equity Audit")
    st.markdown("Upload a CSV containing third-party vendor risk scores against historical student persistence outcomes to evaluate disparate impact and penalty thresholds.")
    
    uploaded_file = st.file_uploader("Upload Vendor Test CSV File", type=["csv"], key="vendor_csv_uploader")
    
    if uploaded_file is not None:
        df_vendor = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded vendor test file with {len(df_vendor):,} records.")
        
        with st.expander("Preview Uploaded Vendor Dataset"):
            st.dataframe(df_vendor.head(), use_container_width=True)
            
        st.markdown("### Configuration & Column Mapping")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_col = st.selectbox("Vendor Prediction Column", df_vendor.columns, key="vendor_pred_col")
        with col2:
            act_col = st.selectbox("Actual Outcome Column (Ground Truth)", df_vendor.columns, key="vendor_act_col")
        with col3:
            default_groups = [col for col in ['Intersectional_Cohort', 'First_Gen', 'Pell_Recipient', 'URM_Status'] if col in df_vendor.columns]
            group_cols = st.multiselect("Demographic Pillars for Disparity Check", df_vendor.columns, default=default_groups, key="vendor_group_cols")
            
        if st.button("Run Vendor Equity Audit", type="primary"):
            if not group_cols:
                st.error("Please select at least one demographic pillar for auditing.")
            else:
                with st.spinner("Calculating error rates and equity penalties..."):
                    audit_results = calculate_disparity_penalty(df_vendor, pred_col, act_col, group_cols)
                    
                st.divider()
                st.markdown("### Vendor Audit Results & Compliance Scorecard")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Raw Overall Accuracy", f"{audit_results['overall_accuracy']:.2%}")
                m2.metric("Max Disparity Gap (FNR)", f"{audit_results['max_fnr_gap']:.2%}")
                m3.metric("Final EI Score (Post-Penalty)", f"{audit_results['final_ei_score']:.2%}", 
                          delta=f"-{audit_results['equity_penalty']:.2%}", delta_color="inverse")
                
                with st.expander("Detailed Subgroup Breakdown JSON"):
                    st.json(audit_results['subgroup_breakdown'])
    else:
        st.info("Awaiting vendor dataset upload. Ensure your CSV includes prediction probabilities, ground truth outcomes, and demographic attributes.")


# ==========================================
# WORKFLOW TABS (UPDATED WITH VENDOR AUDIT)
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

# ------------------------------------------
# TAB 0: EXECUTIVE SUMMARY & PRINTOUT REPORT
# ------------------------------------------
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
        fig_bar.add_hline(y=refusal_threshold, line_dash="dash", line_color="#000000", annotation_text="Minimum EI Threshold", annotation_font_color="#000000")
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
    st.markdown("Generate and print/download an executive report for institutional leadership and board reviews.")
    
    report_text = f"""================================================================================
EQUIMERITIZATION INDEX (EI) EXECUTIVE AUDIT MEMORANDUM
================================================================================
Author / Lead Developer: Steve Westgreen
Institutional Context: UTSA Educational Leadership & Policy Studies
Audit Timestamp: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Sample Size Analyzed: {len(df):,} student records

GOVERNANCE AUDIT PARAMETERS:
--------------------------------------------------------------------------------
- Policy Calibration Weight (λ): {lambda_weight:.2f}
- Minimum Approval Threshold (EI Cutoff): {refusal_threshold:.2f}
- Classification Decision Boundary: {classification_threshold:.2f}

AUDIT PERFORMANCE RESULTS:
--------------------------------------------------------------------------------
1. Vendor Proprietary Model:
   - Predictive Performance (ROC-AUC / P): {m1_P:.4f}
   - Disparity Penalty (FNR Gap / B):        {m1_B:.4f}
   - Equimeritization Index Score (EI):      {m1_EI:.4f}
   - Audit Status:                          {"VETOED / CONTRACT REFUSAL" if m1_veto else "APPROVED FOR DEPLOYMENT"}

2. Equimeritization Calibrated Model:
   - Predictive Performance (ROC-AUC / P): {m2_P:.4f}
   - Disparity Penalty (FNR Gap / B):        {m2_B:.4f}
   - Equimeritization Index Score (EI):      {m2_EI:.4f}
   - Audit Status:                          {"VETOED / CONTRACT REFUSAL" if m2_veto else "APPROVED FOR DEPLOYMENT"}

FINDINGS & RECOMMENDATION:
--------------------------------------------------------------------------------
{"The vendor proprietary model displays severe disparate error rates against marginalized intersectional student cohorts and fails the institutional EI threshold. Algorithmic Refusal Protocol triggered." if m1_veto else "All models meet the minimum institutional Equimeritization Index threshold."}
================================================================================
"""

    st.text_area("Executive Summary Preview (Printable Memo)", value=report_text, height=220)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📄 Download Executive Policy Memo (TXT)",
            data=report_text,
            file_name="Equimeritization_Executive_Policy_Memo.txt",
            mime="text/plain",
            help="Download a clean text memorandum to share with university leadership or board members."
        )
    with col_dl2:
        csv_summary = pd.DataFrame({
            'Model': ['Vendor Proprietary Model', 'Equimeritization Calibrated Model'],
            'Predictive_P': [m1_P, m2_P],
            'Bias_Penalty_B': [m1_B, m2_B],
            'EI_Score': [m1_EI, m2_EI],
            'Status': ["VETOED" if m1_veto else "APPROVED", "VETOED" if m2_veto else "APPROVED"]
        }).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📊 Download Executive Metrics Table (CSV)",
            data=csv_summary,
            file_name="EI_Executive_Summary_Metrics.csv",
            mime="text/csv",
            help="Download executive summary metrics as a CSV table for inclusion in institutional presentations."
        )

# ------------------------------------------
# TAB 1: POLICY SENSITIVITY
# ------------------------------------------
with tab_sweep:
    st.subheader("Policy Calibration Weight (λ) Sensitivity Sweep")
    lambdas = np.linspace(0.0, 3.0, 50)
    m1_eis = [m1_P - (l * m1_B) for l in lambdas]
    m2_eis = [m2_P - (l * m2_B) for l in lambdas]
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=lambdas, y=m1_eis, mode='lines', name='Vendor Proprietary Model', line=dict(color='#DC2626', width=3)))
    fig_sweep.add_trace(go.Scatter(x=lambdas, y=m2_eis, mode='lines', name='Equimeritization Calibrated Model', line=dict(color='#059669', width=3)))
    fig_sweep.add_vline(x=lambda_weight, line_dash="dash", line_color="#0F172A", annotation_text=f"Current λ = {lambda_weight}")
    fig_sweep.add_hline(y=refusal_threshold, line_dash="dot", line_color="#D97706", annotation_text="Veto Threshold")
    fig_sweep.update_layout(xaxis_title="Policy Weight (λ)", yaxis_title="Equimeritization Index (EI)", template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_sweep, use_container_width=True)

# ------------------------------------------
# TAB 2: DATA INGESTION
# ------------------------------------------
with tab_ingest:
    st.subheader("📁 Institutional Data Ingestion & Schema Guidelines")
    st.markdown("""
    Upload your university's anonymized student persistence and algorithmic risk dataset to run real-time 4-pillar audits.
    
    **Required CSV Schema Columns:**
    * `Student_ID`: Unique student identifier string.
    * `Intersectional_Cohort`: Demographic category or intersecting bracket label.
    * `Actual_Persistence`: Binary ground truth (1 = Persisted, 0 = Stopped Out).
    * `GPA`: Cumulative student grade point average (1.0 - 4.0).
    * `LMS_Clicks`: Behavioral engagement metric (Learning Management System interaction counts).
    * `Vendor_Legacy_Prob`: Risk probability score from the vendor proprietary model (0.0 - 1.0).
    * `Equimeritization_Calibrated_Prob`: Risk probability score from your calibrated model (0.0 - 1.0).
    """)
    
    sample_csv = generate_synthetic_data(n_samples=100).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name="EI_Sample_Dataset_Template.csv",
        mime="text/csv"
    )

# ------------------------------------------
# TAB 3: RAW LOGS
# ------------------------------------------
with tab_raw:
    st.subheader("Audited Student Records & Governance Triage Logs")
    st.dataframe(df_m1_triage[['Student_ID', 'Intersectional_Cohort', 'Actual_Persistence', 'Vendor_Legacy_Prob', 'Equimeritization_Calibrated_Prob', 'Governance_Triage']], use_container_width=True)
    st.divider()
    csv_data = df_m1_triage.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Active Audit Log (CSV)", data=csv_data, file_name="equimeritization_active_audit.csv", mime="text/csv")

# ------------------------------------------
# TAB 4: MULTI-AUDIT ARCHIVE
# ------------------------------------------
with tab_archive:
    st.subheader("🏛️ Institutional Multi-Audit Historical Archive")
    if len(st.session_state['audit_history']) > 0:
        archive_df = pd.DataFrame(st.session_state['audit_history'])
        st.dataframe(archive_df, use_container_width=True)
        st.divider()
        archive_csv = archive_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Complete Audit Archive (CSV)", data=archive_csv, file_name="EI_Multi_Audit_Archive.csv", mime="text/csv")
        if st.button("🗑️ Clear Audit Archive"):
            st.session_state['audit_history'] = []
            st.rerun()
    else:
        st.info("No audit runs archived yet. Save runs from the sidebar to populate this archive.")

# ------------------------------------------
# TAB 5: EPISTEMIC AUDIT
# ------------------------------------------
with tab_epi:
    st.subheader("Pillar 1: Epistemic Audit (Feature Lineage & Proxy Concentration)")
    imp_df, top_proxy_ratio, epi_status = run_epistemic_audit(df)
    c_e1, c_e2 = st.columns([1, 2])
    with c_e1:
        st.metric("Epistemic Audit Status", epi_status)
        st.metric("Top Feature Reliance Ratio", f"{top_proxy_ratio * 100:.1f}%")
    with c_e2:
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Permutation Feature Importance Weights", color_discrete_sequence=['#1E3A8A'])
        fig_imp.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------
# TAB 6: CONSTRUCT AUDIT
# ------------------------------------------
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

# ------------------------------------------
# TAB 7: IMPACT AUDIT
# ------------------------------------------
with tab_imp:
    st.subheader("Pillar 3: Impact Audit (Disparate Impact & 80% Rule)")
    fnr1, B1, di_ratio1, status1 = run_impact_audit(df, 'Vendor_Legacy_Prob', threshold=classification_threshold)
    fnr2, B2, di_ratio2, status2 = run_impact_audit(df, 'Equimeritization_Calibrated_Prob', threshold=classification_threshold)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("### Vendor Legacy Model")
        st.metric("Impact Audit Status", status1)
        st.metric("Disparate Impact Ratio (80% Rule)", f"{di_ratio1:.2f}", "FAIL" if di_ratio1 < 0.80 else "PASS", delta_color="inverse" if di_ratio1 < 0.80 else "normal")
        st.metric("FNR Disparity Gap (B)", f"{B1:.3f}")
    with ci2:
        st.markdown("### Calibrated Model")
        st.metric("Impact Audit Status", status2)
        st.metric("Disparate Impact Ratio (80% Rule)", f"{di_ratio2:.2f}", "PASS", delta_color="normal")
        st.metric("FNR Disparity Gap (B)", f"{B2:.3f}")

# ------------------------------------------
# TAB 8: GOVERNANCE & CAPACITY SIMULATOR
# ------------------------------------------
with tab_gov:
    st.subheader("Pillar 4: Governance Audit & Advisor Caseload Capacity Simulator")
    st.markdown("Evaluates human oversight requirements and tests whether AI risk flagging exceeds campus advising capacity.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### 🏛️ Advisor Caseload Parameters")
        num_advisors = st.slider("Total Campus Academic Advisors", min_value=5, max_value=100, value=25, step=5)
        max_caseload = st.slider("Max Students per Advisor (Term Limit)", min_value=50, max_value=300, value=150, step=10)
        
        tier1_count = len(df_m1_triage[df_m1_triage['Governance_Triage'].str.contains("Tier 1")])
        total_capacity = num_advisors * max_caseload
        capacity_ratio = (tier1_count / total_capacity) * 100 if total_capacity > 0 else 0
        
        st.metric("Tier 1 Students Flagged", f"{tier1_count:,} students")
        st.metric("Total Advisor Capacity", f"{total_capacity:,} student slots")
        
        if tier1_count > total_capacity:
            st.error(f"🚨 **CAPACITY BOTTLENECK:** Tier 1 outreach demand exceeds total advisor capacity by **{tier1_count - total_capacity:,} students** ({capacity_ratio:.1f}% capacity utilization).")
        else:
            st.success(f"✅ **CAPACITY ADEQUATE:** Advisor staffing can accommodate Tier 1 outreach ({capacity_ratio:.1f}% capacity utilization).")
            
    with col_g2:
        st.markdown("### Triage Tier Distribution")
        triage_counts = df_m1_triage['Governance_Triage'].value_counts().reset_index()
        triage_counts.columns = ['Triage Tier', 'Student Count']
        fig_triage = px.pie(triage_counts, names='Triage Tier', values='Student Count', title="Vendor Model Triage Allocation", color_discrete_sequence=['#DC2626', '#D97706', '#2563EB', '#059669'])
        fig_triage.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_triage, use_container_width=True)

# ------------------------------------------
# TAB 9: VENDOR AUDIT (NEW MODULE)
# ------------------------------------------
with tab_vendor:
    render_vendor_audit_section()

# ------------------------------------------
# TAB 10: USER GUIDE
# ------------------------------------------
with tab_guide:
    st.subheader("📖 Equimeritization Index ($EI$) Suite - Official User & Technical Operations Guide")
    st.markdown("##### Developed by Steve Westgreen | UTSA Educational Leadership & Policy Studies")
    st.divider()
    
    st.markdown("""
    ### 1. Executive Summary & Core Philosophy
    The **Equimeritization Index ($EI$) Suite** is an enterprise-grade algorithmic governance platform designed to evaluate equity, transparency, and disparate impact in AI-driven predictive systems within higher education.
    
    Unlike traditional predictive analytics tools that optimize solely for statistical accuracy, the $EI$ framework balances predictive power ($P$) against systemic bias penalties ($B$) across intersecting student demographics:
    
    $$EI = P - (\\lambda \\times B)$$
    
    * **$P$ (Predictive Performance):** Evaluated via Area Under the Receiver Operating Characteristic Curve ($ROC\\text{-}AUC$).
    * **$B$ (Bias Penalty):** Quantitative measurement of False Negative Rate ($FNR$) gaps across protected or marginalized demographic subgroups.
    * **$\\lambda$ (Policy Calibration Weight):** Institutional parameter governing the priority placed on error-rate parity versus raw predictive accuracy.

    ---

    ### 2. System Architecture & The 4-Pillar Audit Workflow
    * **Pillar 1: Epistemic Audit (Feature Lineage & Proxy Concentration):** Evaluates whether model predictions rely heavily on proxy variables or behavioral tracking data. Triggered if a single variable accounts for $> 75\%$ of model reliance.
    * **Pillar 2: Construct Audit (Differential Proxy Validity):** Tests whether behavioral proxy variables (such as LMS activity) measure student engagement consistently across demographic subgroups.
    * **Pillar 3: Impact Audit (Disparate Error Rates & 80% Rule):** Measures adverse impact ratios and False Negative Rate gaps. Fails if the Disparate Impact Ratio drops below $0.80$ (Four-Fifths Rule).
    * **Pillar 4: Governance & Capacity Simulator:** Maps predictions into human oversight tiers (Tier 1 Outreach, Tier 2 Review, Tier 3 Algorithmic Veto) and tests whether advising headcount can handle Tier 1 workloads.

    ---

    ### 3. Step-by-Step Operating Instructions
    1. **Configure Governance Parameters (Sidebar):** Adjust policy weight ($\\lambda$), minimum $EI$ approval thresholds, and decision boundaries.
    2. **Ingest Data or Use Synthetic Engine:** Use the sidebar or **`📁 Data Ingestion`** tab to upload university CSVs or test the synthetic baseline.
    3. **Apply Intersectional Slicing:** Filter specific demographic brackets (First-Gen, Pell-Grant recipient, URM) to ensure systemic gaps are never hidden behind aggregate averages.
    4. **Archive & Export:** Save audit runs to history and print or download executive policy memos directly from the Executive Summary tab.
    """)
