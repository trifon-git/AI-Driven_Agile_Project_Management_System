"""
Node2Vec Embedder
=================
Uses torch_geometric's Node2Vec to learn structural embeddings for tasks and developers.
"""
import os
import torch
import torch.nn as nn
import random
from collections import defaultdict
import joblib
from antigrivity.config import MODEL_DIR

CACHE_PATH = os.path.join(MODEL_DIR, "node2vec_embeddings.pkl")

class PureNode2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(num_nodes, embedding_dim)
        self.out_embed = nn.Embedding(num_nodes, embedding_dim)
        # Initialize weights
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.xavier_uniform_(self.out_embed.weight)
        
    def forward(self, center, context, negative):
        in_vecs = self.in_embed(center)        # [batch_size, dim]
        out_vecs = self.out_embed(context)     # [batch_size, dim]
        neg_vecs = self.out_embed(negative)    # [batch_size, num_neg, dim]
        
        pos_out = torch.sum(in_vecs * out_vecs, dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_out)
        
        neg_out = torch.bmm(neg_vecs, in_vecs.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(torch.nn.functional.logsigmoid(-neg_out), dim=1)
        
        return (pos_loss + neg_loss).mean()

class GraphEmbedder:
    def __init__(self, embedding_dim=64, **kwargs):
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def train_embeddings(self, edge_index, num_nodes, epochs=10):
        """
        Train a pure-PyTorch Node2Vec to completely bypass Windows DLL C++ issues.
        """
        print(f"[Node2Vec] Training structural graph embeddings on {num_nodes} nodes (Pure PyTorch)...")
        
        walk_length = self.kwargs.get('walk_length', 15)
        context_size = self.kwargs.get('context_size', 5)
        walks_per_node = self.kwargs.get('walks_per_node', 10)
        num_negative = self.kwargs.get('num_negative_samples', 3)
        
        # 1. Build Adjacency List
        adj = defaultdict(list)
        # Handle both torch tensor and tuple/list edge_index
        if isinstance(edge_index, torch.Tensor):
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
        else:
            src, dst = edge_index
            
        for u, v in zip(src, dst):
            adj[u].append(v)
            adj[v].append(u) # Make undirected for better clustering
            
        # 2. Generate Random Walks
        print(" -> Generating Random Walks...")
        walks = []
        for node in range(num_nodes):
            if len(adj[node]) == 0: continue
            for _ in range(walks_per_node):
                walk = [node]
                curr = node
                for _ in range(walk_length - 1):
                    neighbors = adj[curr]
                    if not neighbors: break
                    curr = random.choice(neighbors)
                    walk.append(curr)
                walks.append(walk)
        
        print(f"Generated {len(walks)} walks")
        
        # 3. Generate Skip-Gram training pairs
        print(" -> Building Training Pairs...")
        centers, contexts = [], []
        for walk in walks:
            for i, center in enumerate(walk):
                # Look behind and ahead in the context window
                start = max(0, i - context_size)
                end = min(len(walk), i + context_size + 1)
                for j in range(start, end):
                    if i != j:
                        centers.append(center)
                        contexts.append(walk[j])
                        
        if not centers:
            # Fallback if graph is empty
            print("⚠️ Insufficient edges for random walks. Returning random embeddings.")
            return torch.randn((num_nodes, self.embedding_dim))
            
        print(f"Training pairs (centers): {len(centers)}")
        
        # 4. DataLoader
        dataset = torch.utils.data.TensorDataset(torch.tensor(centers), torch.tensor(contexts))
        # Use a larger batch size to reduce optimizer steps and stabilize training
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # 5. Model & Optimizer
        model = PureNode2Vec(num_nodes, self.embedding_dim).to(self.device)
        # Lower learning rate for more stable convergence on skip-gram objective
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 6. Training Loop
        print(" -> Training Skip-Gram Model...")
        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for center_b, context_b in loader:
                center_b = center_b.to(self.device)
                context_b = context_b.to(self.device)
                
                # Sample negatives randomly
                neg_b = torch.randint(0, num_nodes, (center_b.size(0), num_negative), device=self.device)
                
                optimizer.zero_grad()
                loss = model(center_b, context_b, neg_b)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            # Print loss every epoch for finer-grained monitoring
            print(f"[Node2Vec] Epoch: {epoch:02d}, Loss: {total_loss / len(loader):.4f}")

        # Return the input embeddings as the structural coordinates
        model.eval()
        with torch.no_grad():
            embeddings = model.in_embed.weight.cpu()
            
        return embeddings

