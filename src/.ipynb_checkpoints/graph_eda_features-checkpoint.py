import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- EDA FUNCTIONS ----------------

def graph_statistics(G: nx.Graph):
    """
    Compute basic graph statistics.
    Returns a dictionary.
    """
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "avg_clustering": nx.average_clustering(G),
        "num_connected_components": nx.number_connected_components(G),
        "graph_diameter": nx.diameter(G.subgraph(max(nx.connected_components(G), key=len)))
    }
    return stats

def plot_degree_distribution(G: nx.Graph, bins: int = 30):
    """
    Plot histogram of node degrees.
    """
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(6,4))
    plt.hist(degrees, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.show()

# ---------------- FEATURE ENGINEERING ----------------

def compute_node_features(G: nx.Graph):
    """
    Compute node-level features and return as a pandas DataFrame:
    - degree
    - degree centrality
    - betweenness centrality
    - closeness centrality
    - eigenvector centrality
    - clustering coefficient
    """
    features = pd.DataFrame(index=G.nodes())

    features["degree"] = dict(G.degree())
    features["degree_centrality"] = nx.degree_centrality(G)
    features["betweenness_centrality"] = nx.betweenness_centrality(G)
    features["closeness_centrality"] = nx.closeness_centrality(G)
    
    try:
        features["eigenvector_centrality"] = nx.eigenvector_centrality(G)
    except nx.NetworkXError:
        features["eigenvector_centrality"] = 0  # fallback for very large graphs

    features["clustering_coefficient"] = nx.clustering(G)

    return features

def save_node_features(features: pd.DataFrame, output_file: str):
    """
    Save node features to CSV.
    """
    features.to_csv(output_file)
    print(f"Node features saved to {output_file}")

# ---------------- MAIN PIPELINE ----------------
if __name__ == "__main__":
    # Example usage: change path to your preprocessed graph
    input_file = "datasets/facebook_cleaned.edgelist"
    output_features = "datasets/facebook_node_features.csv"

    # Load preprocessed graph
    G = nx.read_edgelist(input_file, nodetype=int)

    # EDA
    stats = graph_statistics(G)
    print("Graph Statistics:", stats)
    plot_degree_distribution(G)

    # Feature Engineering
    features = compute_node_features(G)
    print("Sample node features:")
    print(features.head())

    # Save features
    save_node_features(features, output_features)
