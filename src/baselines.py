import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from .config import Config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Simple MLP Class
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1), # Small dropout for stability
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

def run_baselines(train_features, train_labels, test_features, test_labels):
    print("\n⚖️  --- Running Baselines for Comparison ---")
    results = {}
    
    # 1. Nearest Class Mean (NCM)
    print("   1. Training NCM (Nearest Class Mean)...")
    start = time.time()
    ncm = NearestCentroid()
    ncm.fit(train_features, train_labels)
    ncm_preds = ncm.predict(test_features)
    ncm_acc = accuracy_score(test_labels, ncm_preds)
    results['NCM'] = ncm_acc
    print(f"      >> NCM Accuracy: {ncm_acc*100:.2f}% (Time: {time.time()-start:.2f}s)")
    
    # 2. MLP Baseline (Linear Layers + ReLU)
    print("   2. Training MLP (Linear + ReLU)...")
    start = time.time()
    
    # Prepare Data
    device = Config.DEVICE
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_test  = torch.tensor(test_features, dtype=torch.float32).to(device)
    
    # Init Model
    # DINOv2 dimension is 384 for Small, but check actual dim
    input_dim = train_features.shape[1] 
    num_classes = len(np.unique(train_labels))
    
    model = SimpleMLP(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train Loop (Fast: 10 Epochs is usually enough for pre-trained features)
    model.train()
    batch_size = 256
    
    for epoch in range(10): 
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i : i + batch_size]
            batch_X, batch_y = X_train[idx], y_train[idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    mlp_acc = accuracy_score(test_labels, preds)
    results['Linear'] = mlp_acc # Key kept as 'Linear' for compatibility
    print(f"      >> MLP Accuracy: {mlp_acc*100:.2f}% (Time: {time.time()-start:.2f}s)")

    # 2b. Testing MLP on OCCLUDED Data
    print("   2b. Testing MLP on OCCLUDED Data...")
    
    # Simulate Occlusion (Zeroing out last 50% of dimensions)
    # 384 dims -> 6 chunks of 64. Occluding 3 chunks = 192 dims.
    cutoff = input_dim // 2
    X_test_occ = X_test.clone()
    X_test_occ[:, cutoff:] = 0.0 # Zero out top half
    
    model.eval()
    with torch.no_grad():
        logits_occ = model(X_test_occ)
        preds_occ = torch.argmax(logits_occ, dim=1).cpu().numpy()
        
    mlp_acc_occ = accuracy_score(test_labels, preds_occ)
    print(f"      >> MLP Occluded Accuracy: {mlp_acc_occ*100:.2f}%")
    
    return results

def run_statistical_baselines(X_train, y_train, X_test, y_test):
    print("\n📉 --- Running Additional Statistical Baselines ---")
    
    # Ensure CPU numpy
    X_tr = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
    y_tr = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    X_te = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
    y_te = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

    results = {}

    # 1. k-NN (The "Memory Upper Bound")
    # Stores ALL data. TQM should be close to this but efficient.
    print("   Running k-NN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    acc_knn = knn.score(X_te, y_te)
    results['k-NN'] = acc_knn
    print(f"   >> k-NN Accuracy: {acc_knn*100:.2f}%")

    # 2. Deep SLDA (Streaming Linear Discriminant Analysis)
    # The strongest statistical competitor. Assumes 1 Gaussian per class.
    print("   Running Deep SLDA...")
    try:
        slda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        slda.fit(X_tr, y_tr)
        acc_slda = slda.score(X_te, y_te)
        results['SLDA'] = acc_slda
        print(f"   >> SLDA Accuracy: {acc_slda*100:.2f}%")
    except Exception as e:
        print(f"   !! SLDA Failed (Singular Matrix?): {e}")
        results['SLDA'] = 0.0

    return results