import numpy as np
import torch
import torch.nn.functional as F

class StreamingSLDA:
    """
    Streaming Deep Linear Discriminant Analysis (Deep SLDA).
    Maintains class means and a shared within-class covariance matrix incrementally.
    """
    def __init__(self, feature_dim, num_classes, shrinkage=1e-4, device='cuda'):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.shrinkage = shrinkage
        self.device = device
        
        self.class_sums = torch.zeros((num_classes, feature_dim), device=device)
        self.class_counts = torch.zeros(num_classes, device=device)
        self.sum_xxT = torch.zeros((feature_dim, feature_dim), device=device)
        self.total_samples = 0
        
    def update(self, features, labels):
        features = features.to(self.device).float()
        labels = labels.to(self.device).long()
        
        n = features.shape[0]
        self.total_samples += n
        
        # Update sum of outer products
        self.sum_xxT += torch.matmul(features.t(), features)
        
        # Update class sums and counts
        for c in torch.unique(labels):
            mask = labels == c
            self.class_sums[c] += features[mask].sum(dim=0)
            self.class_counts[c] += mask.sum()
            
    def predict(self, features):
        features = features.to(self.device).float()
        
        # Compute means
        valid_classes = self.class_counts > 0
        means = torch.zeros_like(self.class_sums)
        means[valid_classes] = self.class_sums[valid_classes] / self.class_counts[valid_classes].unsqueeze(1)
        
        # Compute shared within-class scatter matrix S_W
        # S_W = sum(x*x^T) - sum_c( N_c * mu_c * mu_c^T )
        mean_outer_sums = torch.zeros_like(self.sum_xxT)
        for c in range(self.num_classes):
            if self.class_counts[c] > 0:
                mu = means[c].unsqueeze(1)
                mean_outer_sums += self.class_counts[c] * torch.matmul(mu, mu.t())
                
        S_W = self.sum_xxT - mean_outer_sums
        
        # Compute Covariance and apply shrinkage
        cov = S_W / max(1, self.total_samples)
        # Shrinkage
        shrinkage_cov = (1 - self.shrinkage) * cov + self.shrinkage * torch.eye(self.feature_dim, device=self.device)
        
        # Precision matrix
        precision = torch.inverse(shrinkage_cov)
        
        # Inference: score = x^T P mu - 0.5 mu^T P mu
        # (N, D) x (D, D) x (D, C) = (N, C)
        term1 = torch.matmul(torch.matmul(features, precision), means.t())
        
        # 0.5 * mu^T P mu for each class -> (C,)
        term2 = 0.5 * torch.sum(torch.matmul(means, precision) * means, dim=1)
        
        scores = term1 - term2
        
        # For unseen classes, set score to -inf
        scores[:, ~valid_classes] = -float('inf')
        
        return torch.argmax(scores, dim=1)
        
    def memory_mb(self):
        # class_sums (C*D), class_counts (C), sum_xxT (D*D)
        bytes_used = (self.num_classes * self.feature_dim + self.num_classes + self.feature_dim ** 2) * 4
        return bytes_used / (1024 ** 2)


class AnalyticCL:
    """
    Analytic Continual Learning (ACL).
    Exact streaming Recursive Least Squares (RLS).
    Tracks R = X^T X + lambda I and Q = X^T Y.
    """
    def __init__(self, feature_dim, num_classes, ridge_lambda=0.1, device='cuda'):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.ridge_lambda = ridge_lambda
        self.device = device
        
        self.R = torch.eye(feature_dim, device=device) * ridge_lambda
        self.Q = torch.zeros((feature_dim, num_classes), device=device)
        
    def update(self, features, labels):
        features = features.to(self.device).float()
        labels = labels.to(self.device).long()
        
        Y = F.one_hot(labels, num_classes=self.num_classes).float()
        
        self.R += torch.matmul(features.t(), features)
        self.Q += torch.matmul(features.t(), Y)
        
    def predict(self, features):
        features = features.to(self.device).float()
        
        # W = R^{-1} Q
        W = torch.linalg.solve(self.R, self.Q)
        
        scores = torch.matmul(features, W)
        return torch.argmax(scores, dim=1)
        
    def memory_mb(self):
        # R (D*D), Q (D*C)
        bytes_used = (self.feature_dim ** 2 + self.feature_dim * self.num_classes) * 4
        return bytes_used / (1024 ** 2)


class RanPAC:
    """
    Random Projection + Analytic Classifier.
    Maps features to a very high dimension (e.g. 10000) using a frozen random matrix,
    then applies RLS.
    """
    def __init__(self, feature_dim, num_classes, projection_dim=10000, ridge_lambda=0.1, device='cuda'):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.ridge_lambda = ridge_lambda
        self.device = device
        
        # Frozen random projection
        self.W_rand = torch.randn((feature_dim, projection_dim), device=device) / (feature_dim ** 0.5)
        
        # RLS States
        self.R = torch.eye(projection_dim, device=device) * ridge_lambda
        self.Q = torch.zeros((projection_dim, num_classes), device=device)
        
    def _project(self, features):
        return F.relu(torch.matmul(features, self.W_rand))
        
    def update(self, features, labels):
        features = features.to(self.device).float()
        labels = labels.to(self.device).long()
        
        Z = self._project(features)
        Y = F.one_hot(labels, num_classes=self.num_classes).float()
        
        self.R += torch.matmul(Z.t(), Z)
        self.Q += torch.matmul(Z.t(), Y)
        
    def predict(self, features):
        features = features.to(self.device).float()
        
        Z = self._project(features)
        
        # W = R^{-1} Q
        W = torch.linalg.solve(self.R, self.Q)
        
        scores = torch.matmul(Z, W)
        return torch.argmax(scores, dim=1)
        
    def memory_mb(self):
        # W_rand (D*M), R (M*M), Q (M*C)
        M = self.projection_dim
        bytes_used = (self.feature_dim * M + M ** 2 + M * self.num_classes) * 4
        return bytes_used / (1024 ** 2)
