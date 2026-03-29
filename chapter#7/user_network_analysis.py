"""
Network Analysis Extension - User Interaction Network

This module extends the user activity analytics with network analysis using NetworkX.
It demonstrates:
1. Building a user interaction network
2. Computing network metrics (centrality, clustering)
3. Identifying influential users through network analysis
4. Community detection in user networks

Libraries used: NetworkX, Pandas, NumPy, Matplotlib
"""

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  NetworkX is not installed. Install with: pip install networkx")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_user_interaction_network(num_users=100, num_interactions=500, random_seed=42):
    """
    Generate a synthetic user interaction network.
    
    Args:
        num_users: Number of unique users
        num_interactions: Number of user interactions
        random_seed: Random seed for reproducibility
        
    Returns:
        networkx.Graph: User interaction network
        pandas.DataFrame: Interaction data
    """
    if not NETWORKX_AVAILABLE:
        print("❌ NetworkX is required for this function")
        return None, None
    
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("  USER INTERACTION NETWORK ANALYSIS WITH NETWORKX")
    print("=" * 70)
    print(f"\n📊 Generating interaction network with:")
    print(f"   - {num_users} unique users")
    print(f"   - {num_interactions} total interactions")
    print()
    
    # Generate interactions (directed edges)
    interactions = []
    for i in range(num_interactions):
        user_from = np.random.randint(1, num_users + 1)
        user_to = np.random.randint(1, num_users + 1)
        
        # Avoid self-loops
        while user_from == user_to:
            user_to = np.random.randint(1, num_users + 1)
        
        interaction_type = np.random.choice(['message', 'like', 'share', 'comment'], 
                                           p=[0.4, 0.3, 0.2, 0.1])
        weight = np.random.randint(1, 10)
        
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 30))
        
        interactions.append({
            'user_from': user_from,
            'user_to': user_to,
            'interaction_type': interaction_type,
            'weight': weight,
            'timestamp': timestamp
        })
    
    df_interactions = pd.DataFrame(interactions)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for user in range(1, num_users + 1):
        G.add_node(user)
    
    # Add weighted edges
    for _, row in df_interactions.iterrows():
        if G.has_edge(row['user_from'], row['user_to']):
            # Increase weight if edge exists
            G[row['user_from']][row['user_to']]['weight'] += row['weight']
        else:
            G.add_edge(row['user_from'], row['user_to'], weight=row['weight'])
    
    print(f"✅ Network generated successfully!")
    print(f"\n📋 Network Statistics:")
    print(f"   - Nodes (Users): {G.number_of_nodes()}")
    print(f"   - Edges (Interactions): {G.number_of_edges()}")
    print(f"   - Network Density: {nx.density(G):.4f}")
    print(f"   - Is Connected: {nx.is_weakly_connected(G)}")
    
    return G, df_interactions


def compute_centrality_metrics(G):
    """
    Compute various centrality metrics for network analysis.
    
    Args:
        G: NetworkX graph
        
    Returns:
        pandas.DataFrame: Centrality metrics for each user
    """
    print("\n" + "=" * 70)
    print("  CENTRALITY METRICS COMPUTATION")
    print("=" * 70)
    
    print("\n🎯 Computing centrality metrics...")
    print("   - Degree Centrality: Measures direct connections")
    print("   - Betweenness Centrality: Measures bridging capability")
    print("   - Closeness Centrality: Measures average distance to others")
    print("   - PageRank: Measures importance based on connections")
    
    # Compute centrality metrics
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, weight='weight')
    
    # Create DataFrame
    centrality_df = pd.DataFrame({
        'user_id': list(G.nodes()),
        'degree_centrality': [degree_centrality[node] for node in G.nodes()],
        'in_degree_centrality': [in_degree_centrality[node] for node in G.nodes()],
        'out_degree_centrality': [out_degree_centrality[node] for node in G.nodes()],
        'betweenness_centrality': [betweenness_centrality[node] for node in G.nodes()],
        'pagerank': [pagerank[node] for node in G.nodes()]
    })
    
    # Compute influence score (weighted combination)
    centrality_df['influence_score'] = (
        centrality_df['pagerank'] * 0.30 +
        centrality_df['betweenness_centrality'] * 0.25 +
        centrality_df['in_degree_centrality'] * 0.25 +
        centrality_df['out_degree_centrality'] * 0.20
    )
    
    # Scale to 0-100
    min_score = centrality_df['influence_score'].min()
    max_score = centrality_df['influence_score'].max()
    centrality_df['influence_score'] = (
        (centrality_df['influence_score'] - min_score) / (max_score - min_score) * 100
    )
    
    print("\n✅ Centrality metrics computed successfully!")
    print("\n📋 Sample centrality metrics (first 5 users):")
    print("-" * 70)
    print(centrality_df.head())
    print("-" * 70)
    
    return centrality_df


def identify_influential_users(centrality_df, top_n=10):
    """
    Identify most influential users based on network metrics.
    
    Args:
        centrality_df: DataFrame with centrality metrics
        top_n: Number of top users to identify
        
    Returns:
        pandas.DataFrame: Top influential users
    """
    print("\n" + "=" * 70)
    print(f"  TOP {top_n} INFLUENTIAL USERS BY NETWORK ANALYSIS")
    print("=" * 70)
    
    # Sort by influence score
    top_users = centrality_df.nlargest(top_n, 'influence_score').copy()
    top_users.insert(0, 'rank', range(1, len(top_users) + 1))
    
    print("\n🏆 Most Influential Users in the Network:")
    print("-" * 70)
    
    display_cols = ['rank', 'user_id', 'influence_score', 'pagerank', 
                   'betweenness_centrality', 'in_degree_centrality']
    print(top_users[display_cols].to_string(index=False))
    print("-" * 70)
    
    print("\n📊 Detailed Network Metrics for Top 5 Users:")
    print("-" * 70)
    
    for idx, row in top_users.head(5).iterrows():
        print(f"\n🥇 Rank #{int(row['rank'])}: User {int(row['user_id'])}")
        print(f"   Influence Score: {row['influence_score']:.2f}/100")
        print(f"   PageRank: {row['pagerank']:.6f}")
        print(f"   Betweenness Centrality: {row['betweenness_centrality']:.6f}")
        print(f"   In-Degree Centrality: {row['in_degree_centrality']:.6f}")
        print(f"   Out-Degree Centrality: {row['out_degree_centrality']:.6f}")
    
    print("\n" + "-" * 70)
    
    return top_users


def detect_communities(G, method='louvain'):
    """
    Detect communities in the user network.
    
    Args:
        G: NetworkX graph
        method: Community detection method
        
    Returns:
        dict: Community assignments
    """
    print("\n" + "=" * 70)
    print("  COMMUNITY DETECTION")
    print("=" * 70)
    
    print(f"\n🔍 Detecting communities using {method} algorithm...")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Use greedy modularity communities (built-in alternative to louvain)
    communities = list(nx.community.greedy_modularity_communities(G_undirected))
    
    # Create community assignment dictionary
    community_assignment = {}
    for i, community in enumerate(communities):
        for node in community:
            community_assignment[node] = i
    
    print(f"\n✅ Communities detected successfully!")
    print(f"   - Number of Communities: {len(communities)}")
    print(f"   - Community Sizes: {[len(c) for c in communities]}")
    
    # Calculate modularity
    modularity = nx.community.modularity(G_undirected, communities)
    print(f"   - Modularity Score: {modularity:.4f}")
    
    return community_assignment, communities


def analyze_community_characteristics(G, community_assignment, df_interactions):
    """
    Analyze characteristics of detected communities.
    
    Args:
        G: NetworkX graph
        community_assignment: Community assignments for nodes
        df_interactions: DataFrame with interaction data
    """
    print("\n" + "=" * 70)
    print("  COMMUNITY CHARACTERISTICS ANALYSIS")
    print("=" * 70)
    
    # Create DataFrame with community assignments
    community_df = pd.DataFrame({
        'user_id': list(community_assignment.keys()),
        'community': list(community_assignment.values())
    })
    
    # Merge with interactions
    df_with_communities = df_interactions.merge(
        community_df, left_on='user_from', right_on='user_id', how='left'
    ).rename(columns={'community': 'community_from'})
    
    df_with_communities = df_with_communities.merge(
        community_df, left_on='user_to', right_on='user_id', how='left', suffixes=('', '_to')
    ).rename(columns={'community': 'community_to'})
    
    # Analyze community statistics
    print("\n📊 Community Statistics:")
    print("-" * 70)
    
    for comm_id in sorted(community_df['community'].unique()):
        comm_users = community_df[community_df['community'] == comm_id]
        comm_size = len(comm_users)
        
        # Get interactions within community
        within_comm = df_with_communities[
            (df_with_communities['community_from'] == comm_id) &
            (df_with_communities['community_to'] == comm_id)
        ]
        
        # Get interactions between communities
        between_comm = df_with_communities[
            (df_with_communities['community_from'] == comm_id) &
            (df_with_communities['community_to'] != comm_id)
        ]
        
        print(f"\n📌 Community {comm_id + 1}:")
        print(f"   - Members: {comm_size} users")
        print(f"   - Internal Interactions: {len(within_comm)}")
        print(f"   - External Interactions: {len(between_comm)}")
        if len(within_comm) + len(between_comm) > 0:
            cohesion = len(within_comm) / (len(within_comm) + len(between_comm)) * 100
            print(f"   - Cohesion: {cohesion:.2f}%")
    
    print("\n" + "-" * 70)


def main():
    """
    Main function to execute network analysis pipeline.
    """
    if not NETWORKX_AVAILABLE:
        print("\n❌ NetworkX is not installed!")
        print("Install with: pip install networkx")
        return
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "NETWORK ANALYSIS - USER INTERACTION NETWORKS" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Step 1: Generate network
    G, df_interactions = generate_user_interaction_network(
        num_users=100, 
        num_interactions=500, 
        random_seed=42
    )
    
    if G is None:
        return
    
    # Step 2: Compute centrality metrics
    centrality_df = compute_centrality_metrics(G)
    
    # Step 3: Identify influential users
    top_users = identify_influential_users(centrality_df, top_n=10)
    
    # Step 4: Detect communities
    community_assignment, communities = detect_communities(G)
    
    # Step 5: Analyze community characteristics
    analyze_community_characteristics(G, community_assignment, df_interactions)
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "ANALYSIS COMPLETED SUCCESSFULLY!" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    return G, centrality_df, top_users, community_assignment


if __name__ == '__main__':
    main()
