import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
def visualize_communities(G, partition, max_nodes=300):
    """
    Visualize graph with nodes color-coded by community.
    
    Args:
        G (nx.Graph): Graph to visualize
        partition (dict): Node -> community mapping
        max_nodes (int): Max nodes to visualize to avoid clutter
    """
    # Subset graph if too large
    if G.number_of_nodes() > max_nodes:
        nodes_subset = list(G.nodes())[:max_nodes]
        G_sub = G.subgraph(nodes_subset)
    else:
        G_sub = G

    # Assign colors based on community
    communities = [partition[n] for n in G_sub.nodes()]
    pos = nx.spring_layout(G_sub, seed=42)  # Force-directed layout

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_size=50, node_color=communities, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    plt.title("Community Structure (Color-coded by Louvain Communities)")
    plt.axis('off')
    plt.show()


def plot_community_size_distribution(partition):
    """
    Plot histogram of community sizes.
    
    Args:
        partition (dict): Node -> community mapping
    """
    df = pd.DataFrame(list(partition.items()), columns=["node", "community"])
    sizes = df.groupby("community").size().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    plt.bar(sizes.index.astype(str), sizes.values, color='skyblue', edgecolor='black')
    plt.xlabel("Community ID")
    plt.ylabel("Number of Nodes")
    plt.title("Community Size Distribution")
    plt.xticks(rotation=90)
    plt.show()
    
    return sizes
