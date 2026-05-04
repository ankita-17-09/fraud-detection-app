import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# TRAIN MODEL FIRST IF NOT EXISTS
if not os.path.exists('model.pkl'):
    st.warning("⚠️ Model not found. Training now... This may take a few minutes.")
    import subprocess
    result = subprocess.run(['python3', 'train_model.py'], capture_output=True, text=True)
    if result.returncode == 0:
        st.success("✅ Model trained successfully!")
    else:
        st.error("❌ Model training failed. Check logs.")
        st.code(result.stderr)

# Page config
st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .fraud-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    .legit-card {
        background: linear-gradient(135deg, #51cf66, #40c057);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# Title
st.title("🔍 AI Fraud Detection System")
st.caption("Powered by Random Forest | Real-time transaction analysis")

# Sidebar
st.sidebar.header("📊 Dashboard")
page = st.sidebar.selectbox("Navigation", ["Single Prediction", "Batch Prediction", "Live Analytics", "Network Graph"])

# Transaction type mapping
type_options = {
    'PAYMENT': {'code': '0', 'risk': 'Low'},
    'TRANSFER': {'code': '1', 'risk': 'High'},
    'CASH_OUT': {'code': '2', 'risk': 'High'},
    'DEBIT': {'code': '3', 'risk': 'Medium'},
    'CASH_IN': {'code': '4', 'risk': 'Low'}
}

def extract_features(amount, oldbalanceOrg, oldbalanceDest, type_code):
    """Extract all 12 features matching training format"""
    return np.array([[
        float(amount),
        float(oldbalanceOrg),
        float(oldbalanceDest),
        int(type_code == '1'),  # is_transfer
        int(type_code == '2'),  # is_cashout
        int(type_code == '3'),  # is_debit
        int(float(oldbalanceDest) == 0),  # receiver_empty
        int(float(oldbalanceOrg) == 0),  # sender_empty
        int(float(oldbalanceOrg) < 0),  # sender_negative
        int(float(amount) > float(oldbalanceOrg)),  # overdraft
        0,  # small_recurring (simplified)
        0   # rapid_transactions (simplified)
    ]])

# ==================== SINGLE PREDICTION PAGE ====================
if page == "Single Prediction":
    st.header("📝 Single Transaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox("💰 Transaction Type", list(type_options.keys()))
        amount = st.number_input("💵 Amount ($)", min_value=0.0, step=100.0, value=1000.0)
        sender_balance = st.number_input("👤 Sender Balance ($)", min_value=0.0, step=100.0, value=5000.0)
    
    with col2:
        receiver_balance = st.number_input("🏦 Receiver Balance ($)", min_value=0.0, step=100.0, value=1000.0)
        sender_id = st.text_input("🆔 Sender ID (optional)", "user_123")
        receiver_id = st.text_input("🆔 Receiver ID (optional)", "merchant_456")
    
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        type_code = type_options[transaction_type]['code']
        
        features = extract_features(amount, sender_balance, receiver_balance, type_code)
        proba = model.predict_proba(features)[0][1]
        is_fraud = proba > 0.5
        
        # HARD-CODED RULES (override model for obvious fraud cases)
        rule_fraud = False
        rule_reasons = []
        
        if amount > sender_balance:
            rule_fraud = True
            rule_reasons.append("⚠️ Amount exceeds sender balance")
        
        if receiver_balance == 0 and amount > 1000:
            rule_fraud = True
            rule_reasons.append("⚠️ Large transfer ($1,000+) to empty account")
        
        if transaction_type in ["TRANSFER", "CASH_OUT"] and amount > 5000 and sender_balance < amount:
            rule_fraud = True
            rule_reasons.append("⚠️ High-risk transfer with insufficient balance")
        
        if sender_balance == 0 and amount > 100:
            rule_fraud = True
            rule_reasons.append("⚠️ Sending from zero-balance account")
        
        # Override if any rule triggered
        if rule_fraud:
            is_fraud = True
            proba = max(proba, 0.95)  # Boost confidence to 95%+
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if is_fraud:
                st.markdown('<div class="fraud-card">', unsafe_allow_html=True)
                st.markdown(f"### 🚨 FRAUD DETECTED")
                st.markdown(f"**Confidence:** {proba*100:.1f}%")
                
                if rule_reasons:
                    st.markdown("**Violations:**")
                    for reason in rule_reasons:
                        st.markdown(f"- {reason}")
                
                st.markdown(f"**Risk Level:** 🔴 CRITICAL")
                st.markdown(f"**Action:** Block transaction immediately")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="legit-card">', unsafe_allow_html=True)
                st.markdown(f"### ✅ LEGITIMATE")
                st.markdown(f"**Confidence:** {(1-proba)*100:.1f}%")
                st.markdown(f"**Risk Level:** 🟢 LOW")
                st.markdown(f"**Action:** Approve transaction")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature breakdown chart
        st.subheader("📊 Risk Factor Breakdown")
        risk_factors = {
            'Amount vs Balance': min(100, (amount / max(sender_balance, 1)) * 50),
            'Empty Receiver': 80 if receiver_balance == 0 else 10,
            'Transfer Type': 70 if type_code in ['1', '2'] else 20,
            'Sender Balance': 60 if sender_balance == 0 else 15
        }
        
        fig2 = px.bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            title="Risk Contribution by Factor",
            color=list(risk_factors.values()),
            color_continuous_scale='Reds',
            labels={'x': 'Risk Score (%)', 'y': ''}
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==================== BATCH PREDICTION PAGE ====================
elif page == "Batch Prediction":
    st.header("📁 Batch Transaction Analysis")
    st.write("Upload a CSV file with columns: `amount`, `oldbalanceOrg`, `oldbalanceDest`, `type`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data")
        st.dataframe(df.head())
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            required = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'type']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                results = []
                for idx, row in df.iterrows():
                    type_str = str(row['type'])
                    type_code = '0'
                    for k, v in type_options.items():
                        if k == type_str:
                            type_code = v['code']
                            break
                    
                    features = extract_features(
                        row['amount'], row['oldbalanceOrg'], 
                        row['oldbalanceDest'], type_code
                    )
                    proba = model.predict_proba(features)[0][1]
                    results.append({
                        'amount': row['amount'],
                        'sender_balance': row['oldbalanceOrg'],
                        'receiver_balance': row['oldbalanceDest'],
                        'type': type_str,
                        'fraud_risk': round(proba * 100, 1),
                        'prediction': 'Fraud' if proba > 0.5 else 'Legitimate'
                    })
                
                result_df = pd.DataFrame(results)
                
                col1, col2, col3 = st.columns(3)
                fraud_count = len(result_df[result_df['prediction'] == 'Fraud'])
                
                with col1:
                    st.metric("Total Transactions", len(result_df))
                with col2:
                    st.metric("Fraud Detected", fraud_count)
                with col3:
                    st.metric("Fraud Rate", f"{(fraud_count/len(result_df))*100:.1f}%")
                
                st.write("### Detailed Results")
                st.dataframe(result_df)
                
                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv, "fraud_predictions.csv", "text/csv")

# ==================== LIVE ANALYTICS PAGE ====================
elif page == "Live Analytics":
    st.header("📈 Live Analytics Dashboard")
    
    # Sample metrics (replace with real data when database is connected)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "12,847", delta="+234")
    with col2:
        st.metric("Fraud Detected", "342", delta="+12", delta_color="inverse")
    with col3:
        st.metric("Fraud Rate", "2.66%", delta="+0.2%", delta_color="inverse")
    with col4:
        st.metric("Model Accuracy", "99.97%", delta="+0.01%")
    
    # Risk distribution pie chart
    st.subheader("🎯 Fraud vs Legitimate Distribution")
    fig_pie = px.pie(
        values=[12505, 342],
        names=['Legitimate', 'Fraud'],
        title="Transaction Status",
        color_discrete_sequence=['#51cf66', '#ff6b6b']
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Time series chart
    st.subheader("📅 Fraud Trends Over Time")
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    fraud_counts = np.random.randint(5, 30, size=30)
    
    fig_line = px.line(
        x=dates, y=fraud_counts,
        title="Daily Fraud Detection",
        labels={'x': 'Date', 'y': 'Fraud Count'}
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Feature importance
    st.subheader("🔑 Top Risk Indicators")
    importance_data = {
        'Feature': ['Transfer Type', 'Overdraft', 'Empty Receiver', 'Amount', 'Cash Out'],
        'Importance': [0.32, 0.25, 0.18, 0.15, 0.10]
    }
    fig_imp = px.bar(importance_data, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

# ==================== NETWORK GRAPH PAGE ====================
elif page == "Network Graph":
    st.header("🕸️ Fraud Network Visualization")
    st.info("This will show connections between suspicious accounts. Coming soon with database integration.")
    
    # Placeholder network visualization
    fig = go.Figure()
    
    # Sample network nodes
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5, 6],
        y=[1, 3, 2, 4, 3, 5],
        mode='markers+text',
        marker=dict(size=[30, 40, 35, 25, 45, 30], color=['red', 'red', 'orange', 'green', 'red', 'orange']),
        text=['Fraudster A', 'Fraudster B', 'Suspicious', 'Clean', 'Fraudster C', 'Suspicious'],
        textposition="top center"
    ))
    
    fig.update_layout(
        title="Account Network Map",
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Legend:**
    - 🔴 Red = Confirmed Fraud
    - 🟠 Orange = Suspicious
    - 🟢 Green = Clean
    """)
# Load model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# Title
st.title("🔍 AI Fraud Detection System")
st.caption("Powered by Random Forest | Real-time transaction analysis")

# Sidebar
st.sidebar.header("📊 Dashboard")
page = st.sidebar.selectbox("Navigation", ["Single Prediction", "Batch Prediction", "Live Analytics", "Network Graph"])

# Transaction type mapping
type_options = {
    'PAYMENT': {'code': '0', 'risk': 'Low'},
    'TRANSFER': {'code': '1', 'risk': 'High'},
    'CASH_OUT': {'code': '2', 'risk': 'High'},
    'DEBIT': {'code': '3', 'risk': 'Medium'},
    'CASH_IN': {'code': '4', 'risk': 'Low'}
}

def extract_features(amount, oldbalanceOrg, oldbalanceDest, type_code):
    """Extract all 12 features matching training format"""
    return np.array([[
        float(amount),
        float(oldbalanceOrg),
        float(oldbalanceDest),
        int(type_code == '1'),  # is_transfer
        int(type_code == '2'),  # is_cashout
        int(type_code == '3'),  # is_debit
        int(float(oldbalanceDest) == 0),  # receiver_empty
        int(float(oldbalanceOrg) == 0),  # sender_empty
        int(float(oldbalanceOrg) < 0),  # sender_negative
        int(float(amount) > float(oldbalanceOrg)),  # overdraft
        0,  # small_recurring (simplified)
        0   # rapid_transactions (simplified)
    ]])

# ==================== SINGLE PREDICTION PAGE ====================
if page == "Single Prediction":
    st.header("📝 Single Transaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox("💰 Transaction Type", list(type_options.keys()))
        amount = st.number_input("💵 Amount ($)", min_value=0.0, step=100.0, value=1000.0)
        sender_balance = st.number_input("👤 Sender Balance ($)", min_value=0.0, step=100.0, value=5000.0)
    
    with col2:
        receiver_balance = st.number_input("🏦 Receiver Balance ($)", min_value=0.0, step=100.0, value=1000.0)
        sender_id = st.text_input("🆔 Sender ID (optional)", "user_123")
        receiver_id = st.text_input("🆔 Receiver ID (optional)", "merchant_456")
    
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        type_code = type_options[transaction_type]['code']
        
        features = extract_features(amount, sender_balance, receiver_balance, type_code)
        proba = model.predict_proba(features)[0][1]
        is_fraud = proba > 0.5
        
        # HARD-CODED RULES (override model for obvious fraud cases)
        rule_fraud = False
        rule_reasons = []
        
        if amount > sender_balance:
            rule_fraud = True
            rule_reasons.append("⚠️ Amount exceeds sender balance")
        
        if receiver_balance == 0 and amount > 1000:
            rule_fraud = True
            rule_reasons.append("⚠️ Large transfer ($1,000+) to empty account")
        
        if transaction_type in ["TRANSFER", "CASH_OUT"] and amount > 5000 and sender_balance < amount:
            rule_fraud = True
            rule_reasons.append("⚠️ High-risk transfer with insufficient balance")
        
        if sender_balance == 0 and amount > 100:
            rule_fraud = True
            rule_reasons.append("⚠️ Sending from zero-balance account")
        
        # Override if any rule triggered
        if rule_fraud:
            is_fraud = True
            proba = max(proba, 0.95)  # Boost confidence to 95%+
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if is_fraud:
                st.markdown('<div class="fraud-card">', unsafe_allow_html=True)
                st.markdown(f"### 🚨 FRAUD DETECTED")
                st.markdown(f"**Confidence:** {proba*100:.1f}%")
                
                if rule_reasons:
                    st.markdown("**Violations:**")
                    for reason in rule_reasons:
                        st.markdown(f"- {reason}")
                
                st.markdown(f"**Risk Level:** 🔴 CRITICAL")
                st.markdown(f"**Action:** Block transaction immediately")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="legit-card">', unsafe_allow_html=True)
                st.markdown(f"### ✅ LEGITIMATE")
                st.markdown(f"**Confidence:** {(1-proba)*100:.1f}%")
                st.markdown(f"**Risk Level:** 🟢 LOW")
                st.markdown(f"**Action:** Approve transaction")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature breakdown chart
        st.subheader("📊 Risk Factor Breakdown")
        risk_factors = {
            'Amount vs Balance': min(100, (amount / max(sender_balance, 1)) * 50),
            'Empty Receiver': 80 if receiver_balance == 0 else 10,
            'Transfer Type': 70 if type_code in ['1', '2'] else 20,
            'Sender Balance': 60 if sender_balance == 0 else 15
        }
        
        fig2 = px.bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            title="Risk Contribution by Factor",
            color=list(risk_factors.values()),
            color_continuous_scale='Reds',
            labels={'x': 'Risk Score (%)', 'y': ''}
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==================== BATCH PREDICTION PAGE ====================
elif page == "Batch Prediction":
    st.header("📁 Batch Transaction Analysis")
    st.write("Upload a CSV file with columns: `amount`, `oldbalanceOrg`, `oldbalanceDest`, `type`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data")
        st.dataframe(df.head())
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            required = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'type']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                results = []
                for idx, row in df.iterrows():
                    type_str = str(row['type'])
                    type_code = '0'
                    for k, v in type_options.items():
                        if k == type_str:
                            type_code = v['code']
                            break
                    
                    features = extract_features(
                        row['amount'], row['oldbalanceOrg'], 
                        row['oldbalanceDest'], type_code
                    )
                    proba = model.predict_proba(features)[0][1]
                    results.append({
                        'amount': row['amount'],
                        'sender_balance': row['oldbalanceOrg'],
                        'receiver_balance': row['oldbalanceDest'],
                        'type': type_str,
                        'fraud_risk': round(proba * 100, 1),
                        'prediction': 'Fraud' if proba > 0.5 else 'Legitimate'
                    })
                
                result_df = pd.DataFrame(results)
                
                col1, col2, col3 = st.columns(3)
                fraud_count = len(result_df[result_df['prediction'] == 'Fraud'])
                
                with col1:
                    st.metric("Total Transactions", len(result_df))
                with col2:
                    st.metric("Fraud Detected", fraud_count)
                with col3:
                    st.metric("Fraud Rate", f"{(fraud_count/len(result_df))*100:.1f}%")
                
                st.write("### Detailed Results")
                st.dataframe(result_df)
                
                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv, "fraud_predictions.csv", "text/csv")

# ==================== LIVE ANALYTICS PAGE ====================
elif page == "Live Analytics":
    st.header("📈 Live Analytics Dashboard")
    
    # Sample metrics (replace with real data when database is connected)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "12,847", delta="+234")
    with col2:
        st.metric("Fraud Detected", "342", delta="+12", delta_color="inverse")
    with col3:
        st.metric("Fraud Rate", "2.66%", delta="+0.2%", delta_color="inverse")
    with col4:
        st.metric("Model Accuracy", "99.97%", delta="+0.01%")
    
    # Risk distribution pie chart
    st.subheader("🎯 Fraud vs Legitimate Distribution")
    fig_pie = px.pie(
        values=[12505, 342],
        names=['Legitimate', 'Fraud'],
        title="Transaction Status",
        color_discrete_sequence=['#51cf66', '#ff6b6b']
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Time series chart
    st.subheader("📅 Fraud Trends Over Time")
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    fraud_counts = np.random.randint(5, 30, size=30)
    
    fig_line = px.line(
        x=dates, y=fraud_counts,
        title="Daily Fraud Detection",
        labels={'x': 'Date', 'y': 'Fraud Count'}
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Feature importance
    st.subheader("🔑 Top Risk Indicators")
    importance_data = {
        'Feature': ['Transfer Type', 'Overdraft', 'Empty Receiver', 'Amount', 'Cash Out'],
        'Importance': [0.32, 0.25, 0.18, 0.15, 0.10]
    }
    fig_imp = px.bar(importance_data, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

# ==================== NETWORK GRAPH PAGE ====================
elif page == "Network Graph":
    st.header("🕸️ Fraud Network Visualization")
    st.info("This will show connections between suspicious accounts. Coming soon with database integration.")
    
    # Placeholder network visualization
    fig = go.Figure()
    
    # Sample network nodes
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5, 6],
        y=[1, 3, 2, 4, 3, 5],
        mode='markers+text',
        marker=dict(size=[30, 40, 35, 25, 45, 30], color=['red', 'red', 'orange', 'green', 'red', 'orange']),
        text=['Fraudster A', 'Fraudster B', 'Suspicious', 'Clean', 'Fraudster C', 'Suspicious'],
        textposition="top center"
    ))
    
    fig.update_layout(
        title="Account Network Map",
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Legend:**
    - 🔴 Red = Confirmed Fraud
    - 🟠 Orange = Suspicious
    - 🟢 Green = Clean
    """)
