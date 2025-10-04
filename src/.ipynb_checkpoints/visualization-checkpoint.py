import os
import networkx as nx
import community as community_louvain
import pandas as pd
import matplotlib.pyplot as plt

def run_louvain_pipeline(G: nx.Graph, output_dir: str = "data/results", visualize_nodes: int = 300):
    """
    Full Louvain community detection pipeline:
    - Detect communities
    - Compute modularity & conductance
    - Compute cluster sizes
    - Save results
    - Generate visualizations

    Args:
        G (nx.Graph): Preprocessed graph
        output_dir (str): Folder to store all outputs
        visualize_nodes (int): Number of nodes to show in graph visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Detect communities
    partition = community_louvain.best_partition(G)
    print(f"Detected {len(set(partition.values()))} communities.")

    # 2️⃣ Compute modularity
    modularity = community_louvain.modularity(partition, G)
    with open(os.path.join(output_dir, "facebook_modularity.txt"), "w") as f:
        f.write(f"Modularity: {modularity:.4f}\n")
    print(f"Modularity saved: {modularity:.4f}")

    # 3️⃣ Save node-community mapping
    df_partition = pd.DataFrame(list(partition.items()), columns=["node", "community"])
    df_partition.to_csv(os.path.join(output_dir, "facebook_communities.csv"), index=False)
    print("Node-community mapping saved.")

    # 4️⃣ Compute and save cluster sizes
    cluster_sizes = df_partition.groupby("community").size().sort_values(ascending=False)
    cluster_sizes.to_csv(os.path.join(output_dir, "facebook_cluster_sizes.csv"), header=["size"])
    print("Cluster size distribution saved.")

    # 5️⃣ Compute average conductance
    conductances = []
    for nodes in df_partition.groupby("community")["node"]:
        nodes_set = set(nodes[1])
        edges_cut = list(nx.edge_boundary(G, nodes_set))
        total_degree = sum(dict(G.degree(nodes_set)).values())
        conductance = len(edges_cut) / total_degree if total_degree > 0 else 0
        conductances.append(conductance)
    avg_conductance = sum(conductances) / len(conductances)
    with open(os.path.join(output_dir, "facebook_conductance.txt"), "w") as f:
        f.write(f"Average Conductance: {avg_conductance:.4f}\n")
    print(f"Average conductance saved: {avg_conductance:.4f}")

    # 6️⃣ Visualize community graph
    if G.number_of_nodes() > visualize_nodes:
        nodes_subset = list(G.nodes())[:visualize_nodes]
        G_sub = G.subgraph(nodes_subset)
    else:
        G_sub = G
    communities_subset = [partition[n] for n in G_sub.nodes()]
    pos = nx.spring_layout(G_sub, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_size=50, node_color=communities_subset, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    plt.title("Louvain Community Structure (subset)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "facebook_community_graph.png"), dpi=300)
    plt.close()
    print("Community graph visualization saved.")

    # 7️⃣ Visualize cluster size distribution
    plt.figure(figsize=(6, 4))
    plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color='skyblue', edgecolor='black')
    plt.xlabel("Community ID")
    plt.ylabel("Number of Nodes")
    plt.title("Cluster Size Distribution")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir, "facebook_cluster_size_distribution.png"), dpi=300)
    plt.close()
    print("Cluster size distribution plot saved.")

    print(f"All results saved in '{output_dir}' folder.")
    return partition, modularity, avg_conductance, cluster_sizes
