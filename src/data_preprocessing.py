import pandas as pd
import networkx as nx

def load_dataset(file_path: str):
    """
    Load a dataset (txt file, space-separated) as an edge list into a pandas DataFrame.
    Works for Facebook, Twitter, or any similar SNAP dataset.
    """
    edges = pd.read_csv(file_path, sep=" ", names=["source", "target"], comment="#")
    print(f"Loaded {len(edges)} edges from {file_path}")
    return edges

def build_graph(edges: pd.DataFrame):
    """
    Build a NetworkX graph from an edge list DataFrame.
    """
    G = nx.from_pandas_edgelist(edges, "source", "target")
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def preprocess_graph(G: nx.Graph):
    """
    Clean the graph for community detection:
    - Remove self-loops
    - Keep the largest connected component
    """
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Keep only largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    print(f"After preprocessing: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def save_graph(G: nx.Graph, output_path: str):
    """
    Save the cleaned graph as an edge list.
    """
    nx.write_edgelist(G, output_path, data=False)
    print(f"Graph saved to {output_path}")

# ---------------- MAIN PIPELINE ----------------
if __name__ == "__main__":
    # Example usage: change input_path and output_path for different datasets
    # Facebook
    fb_input = "datasets/facebook_combined.txt"
    fb_output = "datasets/facebook_cleaned.edgelist"

    edges = load_dataset(fb_input)
    G = build_graph(edges)
    G = preprocess_graph(G)
    save_graph(G, fb_output)

    # Twitter
    tw_input = "datasets/twitter_combined.txt"
    tw_output = "datasets/twitter_cleaned.edgelist"

    edges = load_dataset(tw_input)
    G = build_graph(edges)
    G = preprocess_graph(G)
    save_graph(G, tw_output)
