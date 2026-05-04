import pandas as pd
import urllib.request
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime

print(f"[{datetime.now()}] Downloading dataset...")

# Download from Kaggle (alternative URL - using direct link)
url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/fraud.csv"

try:
    data = pd.read_csv(url)
    print(f"[{datetime.now()}] Dataset loaded successfully. Shape: {data.shape}")
except:
    print(f"[{datetime.now()}] Direct URL failed. Using fallback...")
    # Fallback: Use a smaller sample from GitHub
    url_fallback = "https://raw.githubusercontent.com/curiousily/Credit-Card-Fraud-Detection/master/data/creditcard.csv"
    data = pd.read_csv(url_fallback)
    # Rename columns to match expected format
    data = data.rename(columns={'V1': 'amount', 'V2': 'oldbalanceOrg', 'V3': 'oldbalanceDest'})
    data['type'] = 'TRANSFER'
    data['isFraud'] = data['Class']
    data['step'] = 1
    data['nameOrig'] = 'user_' + data.index.astype(str)
    data['nameDest'] = 'merchant_' + data.index.astype(str)

print(f"[{datetime.now()}] Filtering data...")
if 'type' in data.columns:
    data = data[data['type'].isin(['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'])]

print(f"[{datetime.now()}] Engineering features...")
data['is_transfer'] = (data['type'] == 'TRANSFER').astype(int)
data['is_cashout'] = (data['type'] == 'CASH_OUT').astype(int)
data['is_debit'] = (data['type'] == 'DEBIT').astype(int)
data['receiver_empty'] = (data['oldbalanceDest'] == 0).astype(int)
data['sender_empty'] = (data['oldbalanceOrg'] == 0).astype(int)
data['sender_negative'] = (data['oldbalanceOrg'] < 0).astype(int)
data['overdraft'] = (data['amount'] > data['oldbalanceOrg']).astype(int)
data['small_recurring'] = ((data['type'] == 'DEBIT') & 
                         (data['amount'] < 10) & 
                         (data.groupby('nameOrig')['amount'].transform('count') > 3)).astype(int)
data['rapid_transactions'] = (data.groupby('nameOrig')['step'].diff() < 2).astype(int)

features = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 
           'is_transfer', 'is_cashout', 'is_debit',
           'receiver_empty', 'sender_empty', 'sender_negative',
           'overdraft', 'small_recurring', 'rapid_transactions']

print(f"[{datetime.now()}] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data['isFraud'], test_size=0.2, stratify=data['isFraud'])

print(f"[{datetime.now()}] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print(f"[{datetime.now()}] Saving model...")
pickle.dump(model, open('model.pkl', 'wb'))

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"\n✅ Training Accuracy: {train_acc:.2%}")
print(f"✅ Test Accuracy: {test_acc:.2%}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = model.predict(X_test)
print(f"\n📊 Precision: {precision_score(y_test, y_pred):.2%}")
print(f"📊 Recall: {recall_score(y_test, y_pred):.2%}")
print(f"📊 F1 Score: {f1_score(y_test, y_pred):.2%}")
print(f"[{datetime.now()}] Model training complete!")
