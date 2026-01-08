import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

def run_baselines(train_features, train_labels, test_features, test_labels):
    print("\n⚖️  --- Running Baselines for Comparison ---")
    results = {}
    
    # 1. Nearest Class Mean (NCM)
    # This is the "dumb" version of your Graph. It just averages the vectors.
    # If your Graph beats this, it proves that "Topology > Averaging".
    print("   1. Training NCM (Nearest Class Mean)...")
    start = time.time()
    ncm = NearestCentroid()
    ncm.fit(train_features, train_labels)
    ncm_preds = ncm.predict(test_features)
    ncm_acc = accuracy_score(test_labels, ncm_preds)
    results['NCM'] = ncm_acc
    print(f"      >> NCM Accuracy: {ncm_acc*100:.2f}% (Time: {time.time()-start:.2f}s)")
    
    # 2. Linear Probe (Logistic Regression)
    # This is the standard "Parametric" baseline. 
    # It usually has high clean accuracy but fails on OOD/Robustness.
    print("   2. Training Linear Probe (Logistic Regression)...")
    start = time.time()
    # max_iter=100 is low, but sufficient for a quick check on DINO features
    clf = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200, multi_class='multinomial')
    clf.fit(train_features, train_labels)
    lin_preds = clf.predict(test_features)
    lin_acc = accuracy_score(test_labels, lin_preds)
    results['Linear'] = lin_acc
    print(f"      >> Linear Accuracy: {lin_acc*100:.2f}% (Time: {time.time()-start:.2f}s)")
    
    return results