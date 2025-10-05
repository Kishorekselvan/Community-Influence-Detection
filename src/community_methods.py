import networkx as nx
import community as community_louvain   # python-louvain
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sklearn.cluster import KMeans
from node2vec import Node2Vec

try:
    from infomap import Infomap
except ImportError:
    Infomap = None

# -------------------- Louvain & Metrics --------------------

def louvain_communities(G: nx.Graph, save_path: str = None):
    """
    Apply Louvain community detection.
    """
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)
    print(f"Louvain detected {len(set(partition.values()))} communities with modularity = {modularity:.4f}")

    if save_path:
        save_node_community_mapping(partition, save_path)
    return partition


def compute_modularity(G, partition):
    modularity = community_louvain.modularity(partition, G)
    print(f"Modularity: {modularity:.4f}")
    return modularity


def compute_conductance(G, partition):
    """
    Compute average conductance across communities.
    
    Args:
        G (nx.Graph)
        partition (dict): Node -> community mapping
        
    Returns:
        float: Average conductance
    """
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, set()).add(node)
    
    conductances = []
    for nodes in communities.values():
        nodes_set = set(nodes)
        edges_cut = list(nx.edge_boundary(G, nodes_set))  # Convert generator to list
        total_degree = sum(dict(G.degree(nodes_set)).values())
        conductance = len(edges_cut) / total_degree if total_degree > 0 else 0
        conductances.append(conductance)
    
    avg_conductance = sum(conductances) / len(conductances)
    print(f"Average Conductance: {avg_conductance:.4f}")
    return avg_conductance
def cluster_size_distribution(partition, plot=True):
    df = pd.DataFrame(list(partition.items()), columns=["node", "community"])
    cluster_sizes = df.groupby("community").size().sort_values(ascending=False)
    
    if plot:
        plt.figure(figsize=(6,4))
        plt.hist(cluster_sizes, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Community Size")
        plt.ylabel("Number of Communities")
        plt.title("Cluster Size Distribution")
        plt.show()
    
    print("Cluster size summary:")
    print(cluster_sizes.describe())
    return cluster_sizes

# -------------------- Visualization --------------------

def visualize_louvain(G: nx.Graph, partition: dict, num_nodes: int = 200, save_path: str = None):
    """
    Visualize graph with nodes colored by community.
    """
    if G.number_of_nodes() > num_nodes:
        nodes_subset = list(G.nodes())[:num_nodes]
        G_sub = G.subgraph(nodes_subset)
    else:
        G_sub = G

    communities = [partition[n] for n in G_sub.nodes()]
    pos = nx.spring_layout(G_sub, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_sub, pos, node_size=40, node_color=communities, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    plt.title("Louvain Community Detection (subset)")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
    plt.show()

# -------------------- Saving Results --------------------

def save_node_community_mapping(partition, save_path="results/facebook_communities.csv"):
    """
    Save node-community mapping as CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(list(partition.items()), columns=["node", "community"])
    df.to_csv(save_path, index=False)
    print(f"Node-community mapping saved to {save_path}")


def save_modularity(modularity, save_path="results/facebook_modularity.txt"):
    """
    Save modularity score to a text file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(f"Modularity: {modularity:.4f}\n")
    print(f"Modularity saved to {save_path}")


def save_cluster_sizes(cluster_sizes, save_path="results/facebook_cluster_sizes.csv"):
    """
    Save cluster size distribution to CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cluster_sizes.to_csv(save_path, header=["size"])
    print(f"Cluster size distribution saved to {save_path}")


def save_cluster_size_plot(cluster_sizes, save_path="results/facebook_cluster_size_distribution.png"):
    """
    Save histogram of cluster sizes as PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color='skyblue', edgecolor='black')
    plt.xlabel("Community ID")
    plt.ylabel("Number of Nodes")
    plt.title("Cluster Size Distribution")
    plt.xticks(rotation=90)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Cluster size plot saved to {save_path}")














def infomap_communities(G: nx.Graph, save_path: str = None):
    """Placeholder for Infomap to allow comparison to proceed."""
    print("Infomap is currently SKIPPED due to installation issues.")
    return None, 0.0

def node2vec_kmeans_communities(G: nx.Graph, n_clusters: int, dimensions: int = 64, save_path: str = None):
    """
    Apply Node2Vec for embedding followed by KMeans clustering.
    """
    start_time = time.time()
    
    # 1. Generate Embeddings (Node2Vec)
    G_str = nx.relabel_nodes(G, lambda x: str(x), copy=True) 
    
    node2vec = Node2Vec(G_str, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    nodes = list(G_str.nodes())
    embeddings = [model.wv[node] for node in nodes]
    
    # 2. Cluster Embeddings (KMeans)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 3. Map cluster labels back to a partition dictionary
    partition = {int(node): label for node, label in zip(nodes, cluster_labels)}
    runtime = time.time() - start_time
    
    print(f"Node2Vec + KMeans detected {len(set(partition.values()))} communities. Runtime: {runtime:.2f}s")
    
    if save_path:
        # Assumes save_node_community_mapping is defined in the original file
        save_node_community_mapping(partition, save_path) 
        
    return partition, runtime