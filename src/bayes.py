import torch
import torch.nn as nn
import numpy as np
from .config import Config
from sklearn.cluster import MiniBatchKMeans

class StreamingNaiveBayes(nn.Module):
    def __init__(self, n_chunks, codebooks):
        super().__init__()
        self.n_chunks = n_chunks
        self.device = Config.DEVICE
        
        # 1. Store Codebooks (To convert Features -> Integers during inference)
        # codebooks is a list of numpy arrays (cluster centers)
        self.codebooks = [torch.tensor(cb, dtype=torch.float32).to(self.device) for cb in codebooks]
        
        # 2. Probability Tables
        # We assume max 200 classes for TinyImageNet. 
        # If unknown, we can make this dynamic, but fixed is faster.
        self.n_classes = Config.N_TASKS * Config.CLASSES_PER_TASK
        
        # Class Priors: P(Class) -> How often do we see "Dog"?
        self.register_buffer("class_counts", torch.zeros(self.n_classes, device=self.device))
        
        # Feature Counts: P(Code | Class)
        # Using a Dictionary allows for "Infinite Vocabulary" (Append-Only)
        # Key: (chunk_id, code_id) -> Value: Tensor of shape (n_classes,)
        self.feature_counts = {} 
        self.words_per_chunk = Config.WORDS_PER_TASK
        self.edge_counts = {}

    def quantize(self, features):
        """
        Helper: Convert Raw Features (Batch, 384) -> Integer Codes (Batch, 6)
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(self.device)
        
        chunks = features.chunk(self.n_chunks, dim=1)
        batch_codes = []
        
        for c in range(self.n_chunks):
            # (Batch, D) @ (K, D).T -> (Batch, K)
            sims = torch.matmul(chunks[c], self.codebooks[c].t())
            # Hard assignment (Top-1)
            codes = torch.argmax(sims, dim=1)
            batch_codes.append(codes)
            
        return torch.stack(batch_codes, dim=1) # (Batch, n_chunks)

    def partial_fit(self, codes, labels):
        """
        Update probability tables.
        codes: (Batch, n_chunks) Integers
        labels: (Batch,) Integers
        """
        codes = codes.to(self.device)
        labels = labels.to(self.device)
        
        # 1. Update Class Priors
        unique_lbls, counts = labels.unique(return_counts=True)
        self.class_counts[unique_lbls] += counts.float()
        
        # 2. Update Feature Likelihoods
        # Iterate over chunks
        for c in range(self.n_chunks):
            chunk_codes = codes[:, c]
            
            # Find unique (code, label) pairs to perform sparse updates
            # Stack: [Code, Label]
            pairs = torch.stack([chunk_codes, labels], dim=1)
            unique_pairs, pair_counts = pairs.unique(dim=0, return_counts=True)
            
            for i in range(len(unique_pairs)):
                code = unique_pairs[i, 0].item()
                lbl = unique_pairs[i, 1].item()
                count = pair_counts[i].item()
                
                key = (c, code)
                if key not in self.feature_counts:
                    self.feature_counts[key] = torch.zeros(self.n_classes, device=self.device)
                
                self.feature_counts[key][lbl] += count

        self._update_edges(codes)

    def _update_edges(self, batch_codes):
        """
        Learns the topology: "Fire together, wire together."
        """
        batch_size = batch_codes.shape[0]
        
        # Convert local codes to Global IDs
        # Chunk 0: 0-511, Chunk 1: 512-1023, etc.
        offsets = torch.arange(self.n_chunks, device=self.device) * self.words_per_chunk
        global_codes = batch_codes + offsets.unsqueeze(0) # (Batch, n_chunks)
        
        # We only track edges between DIFFERENT chunks (Chain or All-to-All)
        # All-to-All allows "Head" to predict "Tail" even if "Body" is missing.
        # There are (6 choose 2) = 15 pairs of chunks.
        
        # Efficient Vectorized Update using Unique Pairs
        global_codes_cpu = global_codes.cpu().numpy()
        
        for i in range(self.n_chunks):
            for j in range(i + 1, self.n_chunks):
                # Get the pairs for this chunk combination
                col_i = global_codes_cpu[:, i]
                col_j = global_codes_cpu[:, j]
                
                # Stack and count unique pairs to avoid loop overhead
                # We use a simple numpy trick for counting pairs
                # (Can be optimized further with sparse matrices but dict is safe)
                
                # Zip into list of tuples
                pairs = list(zip(col_i, col_j))
                
                # Naive loop is actually fast enough for batch_size=256
                # Optimized: simple aggregation
                for u, v in pairs:
                    if u > v: u, v = v, u # Sort for undirected edge
                    key = (u, v)
                    self.edge_counts[key] = self.edge_counts.get(key, 0) + 1
    
    def diffuse_graph(self, alpha=0.2):
        """
        Graph Smoothing: Propagates class probability mass across edges.
        Call this ONCE after training, before evaluation.
        
        alpha: How much 'trust' to give to neighbors (0.0 - 1.0)
        """
        print(f"🕸️  Applying Hebbian Graph Diffusion (alpha={alpha})...")
        
        # We need to add "Phantom Counts" to nodes based on their neighbors.
        # This is essentially: P_new(C|u) = P(C|u) + alpha * sum(P(C|v) * weight_uv)
        
        updates = {}
        
        # Iterate through all learned edges
        # key: (u, v), val: edge_strength
        for (u, v), count in self.edge_counts.items():
            if count < 3: continue # Prune weak edges (noise)
            
            # Convert Global ID back to (Chunk, Local)
            c_u, l_u = divmod(u, self.words_per_chunk)
            c_v, l_v = divmod(v, self.words_per_chunk)
            
            key_u = (c_u, l_u)
            key_v = (c_v, l_v)
            
            # If one node has knowledge, share it with the other
            has_u = key_u in self.feature_counts
            has_v = key_v in self.feature_counts
            
            if has_u:
                vec_u = self.feature_counts[key_u]
                # Prepare update for V
                if key_v not in updates: updates[key_v] = torch.zeros_like(vec_u)
                # Weight by edge count? Normalize? 
                # Simple version: Fixed alpha * normalized_source
                updates[key_v] += (vec_u * alpha * 0.1) # 0.1 is a dampener
                
            if has_v:
                vec_v = self.feature_counts[key_v]
                # Prepare update for U
                if key_u not in updates: updates[key_u] = torch.zeros_like(vec_v)
                updates[key_u] += (vec_v * alpha * 0.1)
                
        # Apply updates
        for key, vec in updates.items():
            if key not in self.feature_counts:
                self.feature_counts[key] = vec
            else:
                self.feature_counts[key] += vec
                
        print(f"✅ Graph Diffusion Complete. Topology integrated.")

    def predict(self, input_features, mask=None, mode='soft', top_k=15, temp=0.5):
        """
        Soft-Weighted Bayes Inference.
        Instead of 1 word per chunk, we let the Top-K nearest words vote.
        
        top_k: Number of neighbors to consult (e.g., 5).
        temp: Temperature for softmax. Lower = sharper weights.
        """
        input_features = input_features.to(self.device)
        chunks = input_features.chunk(self.n_chunks, dim=1)
        batch_sz = input_features.shape[0]
        
        # 1. Log Priors (Batch, Classes)
        total_docs = self.class_counts.sum() + 1e-9
        # P(C)
        class_log_probs = torch.log(self.class_counts + 1) - torch.log(total_docs + self.n_classes)
        log_probs = class_log_probs.unsqueeze(0).repeat(batch_sz, 1) 
        
        # 2. Accumulate Evidence from Top-K Words
        for c in range(self.n_chunks):
            if mask is not None and not mask[c]: continue
            
            # --- A. Get Top-K Nearest Words ---
            # (Batch, 64) @ (Vocab, 64).T -> (Batch, Vocab)
            sims = torch.matmul(chunks[c], self.codebooks[c].t())
            
            # Get values and indices of top-k matches
            # vals: (Batch, K), indices: (Batch, K)
            k_vals, k_indices = torch.topk(sims, k=top_k, dim=1)
            
            # --- B. Convert Similarity to Attention Weights ---
            # We want weights that sum to 1.0 for the K neighbors
            # Softmax: shape (Batch, K)
            attn_weights = torch.softmax(k_vals / temp, dim=1)
            
            # --- C. Weighted Voting ---
            # We need to look up probabilities for ALL K neighbors efficiently.
            # Since standard dict lookup is slow, we loop over K (small number, e.g. 5)
            
            # Buffer for this chunk's votes: (Batch, Classes)
            chunk_votes = torch.zeros(batch_sz, self.n_classes, device=self.device)
            
            for k in range(top_k):
                # Get the k-th best code for every image in batch
                # shape: (Batch,)
                k_codes = k_indices[:, k]
                k_weights = attn_weights[:, k] # (Batch,)
                
                # We loop through batch for dictionary lookup (safest for sparse dicts)
                # (Optimization: In C++/CUDA we would scatter/gather, but Python loop is okay for Batch=100)
                for b in range(batch_sz):
                    code = k_codes[b].item()
                    w = k_weights[b].item()
                    
                    key = (c, code)
                    
                    if key in self.feature_counts:
                        # P(Word | Class)
                        counts = self.feature_counts[key]
                        num = counts + 1
                        den = self.class_counts + 500 # Approx Vocab Size
                        
                        cond_prob = torch.log(num) - torch.log(den)
                        
                        # Add weighted vote
                        chunk_votes[b] += (cond_prob * w)
                    else:
                        # Unseen penalty
                        penalty = torch.log(torch.tensor(1.0)) - torch.log(self.class_counts + 500)
                        chunk_votes[b] += (penalty * w)
            
            # Add this chunk's consensus to total
            log_probs += chunk_votes
                    
        # return torch.argmax(log_probs, dim=1)
        return torch.argmax(self.predict_logits(input_features, mask, top_k, temp), dim=1)
    
    def predict_logits(self, input_features, mask=None, top_k=5, temp=0.1):
        """
        Returns raw log-probabilities instead of class labels.
        Required for the Hybrid (Dual-System) inference.
        """
        input_features = input_features.to(self.device)
        chunks = input_features.chunk(self.n_chunks, dim=1)
        batch_sz = input_features.shape[0]
        
        # 1. Start with Log Priors (Baseline probability of each class)
        # P(C)
        total_docs = self.class_counts.sum() + 1e-9
        class_log_probs = torch.log(self.class_counts + 1) - torch.log(total_docs + self.n_classes)
        log_probs = class_log_probs.unsqueeze(0).repeat(batch_sz, 1) 
        
        # 2. Accumulate Evidence from Top-K Words
        for c in range(self.n_chunks):
            if mask is not None and not mask[c]: continue
            
            # --- A. Get Top-K Nearest Words ---
            sims = torch.matmul(chunks[c], self.codebooks[c].t())
            
            # Get values and indices of top-k matches
            k_vals, k_indices = torch.topk(sims, k=top_k, dim=1)
            
            # --- B. Attention Weights ---
            attn_weights = torch.softmax(k_vals / temp, dim=1)
            
            # --- C. Weighted Voting ---
            chunk_votes = torch.zeros(batch_sz, self.n_classes, device=self.device)
            
            for k in range(top_k):
                k_codes = k_indices[:, k]
                k_weights = attn_weights[:, k]
                
                # Optimized loop for batch processing
                for b in range(batch_sz):
                    code = k_codes[b].item()
                    w = k_weights[b].item()
                    
                    key = (c, code)
                    
                    if key in self.feature_counts:
                        # P(Word | Class)
                        counts = self.feature_counts[key]
                        # Laplace Smoothing (+1)
                        num = counts + 1
                        den = self.class_counts + 500 # Approx Vocab Size per chunk
                        
                        cond_prob = torch.log(num) - torch.log(den)
                        
                        chunk_votes[b] += (cond_prob * w)
                    else:
                        # Penalty for unseen words
                        penalty = torch.log(torch.tensor(1.0)) - torch.log(self.class_counts + 500)
                        chunk_votes[b] += (penalty * w)
            
            # Add this chunk's votes to total
            log_probs += chunk_votes
                    
        # --- RETURN RAW LOGITS (The Change) ---
        return log_probs