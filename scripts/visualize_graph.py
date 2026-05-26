"""
Interactive Graph Visualization
===============================
Uses PyVis to create an interactive HTML visualization of the 
Agile Team/Historical Task graph.

Requirements:
    pip install pyvis
"""

import os
import sys
import networkx as nx
from pyvis.network import Network

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antigrivity.models.graph_builder import AllocationGraphBuilder
from antigrivity.config import TEAM

def visualize_sprint_network():
    print("🕸️ Building Bi-Partite Team Network for visualization...")
    
    # 1. Gather Graph Data
    builder = AllocationGraphBuilder()
    # Build only from history for now (pass empty active tasks list)
    edge_index_tensor, node_to_id, id_to_node, node_data = builder.build_from_history_and_active([])
    
    # 2. Convert to NetworkX for visualization formatting
    G = nx.Graph()
    
    # Add Nodes
    for node_name, node_id in node_to_id.items():
        data = node_data.get(node_id, {})
        summary = data.get("summary", "")
        project = data.get("project", "")
        
        if node_name.startswith("DEV_"):
            clean_name = node_name.replace("DEV_", "")
            color = "#ff6b6b" # Red for developers
            size = 30
            group = "Developer"
            title = f"Developer: {clean_name}"
        else:
            clean_name = node_name.replace("TASK_", "").replace("ACTIVE_", "")
            color = "#4dabf7" # Blue for tasks
            size = 15
            group = "Task"
            # Label shows Key + Summary (truncated if too long for node label, but full in tooltip)
            display_summary = f": {summary}" if summary else ""
            label = f"{clean_name}{display_summary}"
            if len(label) > 100:
                label = label[:97] + "..."
            
            title = f"Task: {clean_name}\nProject: {project}\nSummary: {summary}"
            clean_name = label # Use the combined label
            
        G.add_node(node_id, label=clean_name, title=title, color=color, size=size, group=group)

    # Add Edges
    # edge_index is a tensor of shape (2, N)
    edge_list = edge_index_tensor.t().tolist()
    for src, dst in edge_list:
        G.add_edge(src, dst)

    # 3. Create PyVis Network
    print("🎨 Rendering interactive HTML...")
    net = Network(height="1080px", width="1920px", bgcolor="#949494", font_color="black", notebook=False)
    
    # Load NetworkX graph into PyVis
    net.from_nx(G)
    
    # Physics for cool movement
    net.toggle_physics(True)
    net.barnes_hut()
    
    # Save the file
    output_path = os.path.join("OUTPUT", "sprint_network_viz.html")
    os.makedirs("OUTPUT", exist_ok=True)
    net.save_graph(output_path)
    
    print(f"\n✅ Visualization successful!")
    print(f"    Open this file in your browser: {os.path.abspath(output_path)}")

def visualize_sprint_network_plt(G):
    import matplotlib.pyplot as plt
    print("📉 Reverting to Matplotlib for static visualization...")
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = [data['color'] for n, data in G.nodes(data=True)]
    nx.draw_networkx(G, pos, with_labels=True, node_color=node_colors, 
                     node_size=500, font_size=8, font_color="white", 
                     edge_color="gray", alpha=0.7)
    
    plt.title("Agile Sprint Network (Devs & Historical Tasks)")
    plt.axis('off')
    output_path = os.path.join("OUTPUT", "sprint_network_static.png")
    plt.savefig(output_path, facecolor='#222222')
    print(f"✅ Static visualization saved to: {output_path}")

if __name__ == "__main__":
    try:
        # Build graph
        builder = AllocationGraphBuilder()
        edge_index_tensor, node_to_id, id_to_node, node_data = builder.build_from_history_and_active([])
        G = nx.Graph()
        for node_name, node_id in node_to_id.items():
            color = "#ff6b6b" if node_name.startswith("DEV_") else "#4dabf7"
            # Get summary from node_data
            data = node_data.get(node_id, {})
            summary = data.get("summary", "")
            display_name = node_name.replace("DEV_", "").replace("TASK_", "").replace("ACTIVE_", "")
            if summary:
                display_name = f"{display_name}: {summary[:50]}" # Trunacte for static viz
            G.add_node(node_id, label=display_name, color=color)
        edge_list = edge_index_tensor.t().tolist()
        for src, dst in edge_list:
            G.add_edge(src, dst)

        try:
            visualize_sprint_network()
        except (ImportError, Exception):
            visualize_sprint_network_plt(G)
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
