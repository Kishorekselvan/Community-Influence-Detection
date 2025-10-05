import networkx as nx
import pandas as pd
import time
import os

def compute_influence_metrics(G: nx.Graph, save_path: str = None):
    """
    Computes PageRank, Eigenvector, and Betweenness Centrality.
    """
    
    # 1. PageRank
    print("Computing PageRank...")
    start_pr = time.time()
    pagerank_scores = nx.pagerank(G)
    time_pr = time.time() - start_pr

    # 2. Eigenvector Centrality
    print("Computing Eigenvector Centrality...")
    start_ev = time.time()
    # Increased max_iter for stability, especially in large graphs
    eigenvector_scores = nx.eigenvector_centrality(G, max_iter=1000) 
    time_ev = time.time() - start_ev

    # 3. Betweenness Centrality (Measures control over information flow)
    print("Computing Betweenness Centrality...")
    start_bc = time.time()
    betweenness_scores = nx.betweenness_centrality(G) 
    time_bc = time.time() - start_bc
    
    # 4. Combine and Save
    df_influence = pd.DataFrame({
        'PageRank': pd.Series(pagerank_scores),
        'Eigenvector_Centrality': pd.Series(eigenvector_scores),
        'Betweenness_Centrality': pd.Series(betweenness_scores)
    })
    
    print(f"Metrics computed. Runtimes: PageRank={time_pr:.2f}s, Eigenvector={time_ev:.2f}s, Betweenness={time_bc:.2f}s")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_influence.to_csv(save_path)
        print(f"Influence scores saved to {save_path}")
        
    return df_influence


def get_top_k_influencers(df_influence: pd.DataFrame, k: int = 10):
    """
    Identifies and prints the top k influential nodes for each metric.
    """
    top_k_results = {}
    for metric in df_influence.columns:
        top_k = df_influence.sort_values(by=metric, ascending=False).head(k)
        print(f"\n--- Top {k} by {metric} ---")
        print(top_k[metric])
        top_k_results[metric] = top_k
    return top_k_results