import os
import sys
import torch
import numpy as np
import random
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antigrivity.config import MODEL_DIR
from antigrivity.models.graph_builder import AllocationGraphBuilder
from antigrivity.models.node2vec_embedder import GraphEmbedder

def set_deterministic_seed(seed=42):
    """Pin the random seed for reproducible GNN thesis results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_offline_preprocessing():
    print("🚀 Initializing Offline GNN Preprocessing...")

    # 1. Build Historical Graph
    print(f"\n[1] Constructing Bipartite Graph from historical Jira data...")
    builder = AllocationGraphBuilder()
    # We pass an empty list of running tasks because we ONLY want history
    edge_index, node_to_id, id_to_node, node_data = builder.build_from_history_and_active([])

    print(f"    Graph compiled: {len(node_to_id)} unique nodes.")

    # 2. Train Node2Vec Embeddings
    print(f"\n[2] Training Node2Vec Structural Embeddings...")
    embedder = GraphEmbedder(
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True
    )
    
    epochs = 40
    print(f"Generated nodes: {len(node_to_id)}, training epochs: {epochs}")
    
    # Train and retrieve the static embeddings tensor
    embeddings = embedder.train_embeddings(edge_index, num_nodes=len(node_to_id), epochs=epochs)
    
    print(f"    Training complete! Matrix shape: {embeddings.shape}")

    # 3. Export to PKL
    output_path = os.path.join(MODEL_DIR, "historical_graph_state.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    data = {
        "node_to_id": node_to_id,
        "embeddings": embeddings.cpu().numpy(),  # Save as numpy array for instant CPU loading
        "hyperparams": {
            "dim": 64,
            "seed": 42
        }
    }
    
    joblib.dump(data, output_path)
    print(f"\n✅ Offline Preprocessing Successful.")
    print(f"    Saved static graph state to: {output_path}")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    run_offline_preprocessing()
