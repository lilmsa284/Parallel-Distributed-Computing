"""
Big Data Analytics - User Activity Session Analysis

This module demonstrates:
1. Synthetic dataset generation for user activity sessions
2. Data filtering and aggregation
3. Session-level statistics computation
4. Top users identification based on performance metrics

Libraries used: Pandas, NumPy, Scikit-learn
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def generate_synthetic_dataset(num_users=100, num_sessions=1000, random_seed=42):
    """
    Generate synthetic user activity session data.
    
    Args:
        num_users: Number of unique users
        num_sessions: Total number of sessions to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Synthetic dataset with user activity sessions
    """
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("  SYNTHETIC DATASET GENERATION - USER ACTIVITY SESSIONS")
    print("=" * 70)
    print(f"\n📊 Generating dataset with:")
    print(f"   - {num_users} unique users")
    print(f"   - {num_sessions} total sessions")
    print()
    
    # Generate user IDs
    user_ids = np.random.randint(1, num_users + 1, num_sessions)
    
    # Generate session data
    session_data = {
        'user_id': user_ids,
        'session_id': [f'SESSION_{i:05d}' for i in range(1, num_sessions + 1)],
        'session_duration_minutes': np.random.exponential(scale=30, size=num_sessions),
        'pages_viewed': np.random.poisson(lam=10, size=num_sessions),
        'clicks': np.random.poisson(lam=15, size=num_sessions),
        'transactions': np.random.binomial(n=1, p=0.15, size=num_sessions),
        'revenue': np.random.lognormal(mean=3.5, sigma=1.0, size=num_sessions),
        'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], 
                                       size=num_sessions, 
                                       p=[0.5, 0.4, 0.1]),
        'bounce': np.random.binomial(n=1, p=0.3, size=num_sessions)
    }
    
    # Generate timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    ) for _ in range(num_sessions)]
    
    session_data['timestamp'] = timestamps
    
    # Create DataFrame
    df = pd.DataFrame(session_data)
    
    # Adjust revenue for transactions
    df.loc[df['transactions'] == 0, 'revenue'] = 0
    
    print(f"✅ Dataset generated successfully!")
    print(f"\n📋 Dataset preview (first 5 rows):")
    print("-" * 70)
    print(df.head())
    print("-" * 70)
    
    return df


def filter_and_aggregate_sessions(df):
    """
    Filter and aggregate session data to compute session-level statistics.
    
    Args:
        df: Input DataFrame with session data
        
    Returns:
        pandas.DataFrame: Aggregated statistics per user
    """
    print("\n" + "=" * 70)
    print("  DATA FILTERING AND AGGREGATION")
    print("=" * 70)
    
    # Filter out bounced sessions for engagement analysis
    engaged_sessions = df[df['bounce'] == 0].copy()
    
    print(f"\n🔍 Filtering bounced sessions:")
    print(f"   - Total sessions: {len(df)}")
    print(f"   - Bounced sessions: {len(df[df['bounce'] == 1])}")
    print(f"   - Engaged sessions: {len(engaged_sessions)}")
    
    # Compute user-level aggregations
    print(f"\n📊 Computing session-level statistics per user...")
    
    user_stats = df.groupby('user_id').agg({
        'session_id': 'count',  # Total sessions
        'session_duration_minutes': ['mean', 'sum'],  # Average and total duration
        'pages_viewed': ['mean', 'sum'],  # Average and total pages
        'clicks': ['mean', 'sum'],  # Average and total clicks
        'transactions': 'sum',  # Total transactions
        'revenue': 'sum',  # Total revenue
        'bounce': 'mean'  # Bounce rate
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = [
        'user_id', 'total_sessions', 
        'avg_session_duration', 'total_session_duration',
        'avg_pages_viewed', 'total_pages_viewed',
        'avg_clicks', 'total_clicks',
        'total_transactions', 'total_revenue',
        'bounce_rate'
    ]
    
    # Compute additional metrics
    user_stats['engagement_score'] = (
        user_stats['avg_pages_viewed'] * 0.3 +
        user_stats['avg_clicks'] * 0.3 +
        user_stats['avg_session_duration'] * 0.2 +
        (1 - user_stats['bounce_rate']) * 50 * 0.2
    )
    
    user_stats['conversion_rate'] = (
        user_stats['total_transactions'] / user_stats['total_sessions'] * 100
    )
    
    user_stats['avg_revenue_per_session'] = (
        user_stats['total_revenue'] / user_stats['total_sessions']
    )
    
    print(f"\n✅ Aggregation completed!")
    print(f"   - Total unique users: {len(user_stats)}")
    print(f"\n📋 Aggregated statistics preview (first 5 users):")
    print("-" * 70)
    print(user_stats.head())
    print("-" * 70)
    
    return user_stats


def compute_performance_metric(user_stats):
    """
    Compute a comprehensive performance metric for each user.
    
    Args:
        user_stats: DataFrame with user-level statistics
        
    Returns:
        pandas.DataFrame: User statistics with performance score
    """
    print("\n" + "=" * 70)
    print("  PERFORMANCE METRIC CALCULATION")
    print("=" * 70)
    
    print("\n🎯 Computing comprehensive performance score...")
    print("   Performance metric combines:")
    print("   - Engagement score (30%)")
    print("   - Total revenue (30%)")
    print("   - Conversion rate (20%)")
    print("   - Total sessions (20%)")
    
    # Normalize metrics using StandardScaler
    scaler = StandardScaler()
    
    # Select features for normalization
    features = ['engagement_score', 'total_revenue', 'conversion_rate', 'total_sessions']
    normalized_features = scaler.fit_transform(user_stats[features])
    
    # Compute weighted performance score
    weights = [0.30, 0.30, 0.20, 0.20]  # Weights for each feature
    performance_scores = np.dot(normalized_features, weights)
    
    # Add performance score to dataframe
    user_stats['performance_score'] = performance_scores
    
    # Scale to 0-100 range for better interpretability
    min_score = user_stats['performance_score'].min()
    max_score = user_stats['performance_score'].max()
    user_stats['performance_score'] = (
        (user_stats['performance_score'] - min_score) / (max_score - min_score) * 100
    )
    
    print("\n✅ Performance scores calculated successfully!")
    
    return user_stats


def identify_top_users(user_stats, top_n=10):
    """
    Identify and display top users based on performance metric.
    
    Args:
        user_stats: DataFrame with user statistics and performance scores
        top_n: Number of top users to display
        
    Returns:
        pandas.DataFrame: Top users by performance score
    """
    print("\n" + "=" * 70)
    print(f"  TOP {top_n} USERS BY PERFORMANCE SCORE")
    print("=" * 70)
    
    # Sort by performance score
    top_users = user_stats.nlargest(top_n, 'performance_score').copy()
    
    # Add rank
    top_users.insert(0, 'rank', range(1, len(top_users) + 1))
    
    print("\n🏆 Top Performing Users:")
    print("-" * 70)
    
    # Display key metrics for top users
    display_cols = [
        'rank', 'user_id', 'performance_score', 'total_sessions',
        'engagement_score', 'total_revenue', 'conversion_rate'
    ]
    
    print(top_users[display_cols].to_string(index=False))
    print("-" * 70)
    
    # Display detailed statistics
    print("\n📊 Detailed Statistics for Top Users:")
    print("-" * 70)
    
    for idx, row in top_users.head(5).iterrows():
        print(f"\n🥇 Rank #{int(row['rank'])}: User {int(row['user_id'])}")
        print(f"   Performance Score: {row['performance_score']:.2f}/100")
        print(f"   Total Sessions: {int(row['total_sessions'])}")
        print(f"   Total Revenue: ${row['total_revenue']:.2f}")
        print(f"   Engagement Score: {row['engagement_score']:.2f}")
        print(f"   Conversion Rate: {row['conversion_rate']:.2f}%")
        print(f"   Avg Session Duration: {row['avg_session_duration']:.2f} minutes")
        print(f"   Bounce Rate: {row['bounce_rate']*100:.2f}%")
    
    print("\n" + "-" * 70)
    
    return top_users


def display_summary_statistics(df, user_stats):
    """
    Display summary statistics for the entire dataset.
    
    Args:
        df: Original session DataFrame
        user_stats: User-level statistics DataFrame
    """
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\n📈 Dataset Overview:")
    print(f"   Total Sessions: {len(df)}")
    print(f"   Unique Users: {df['user_id'].nunique()}")
    print(f"   Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    print(f"\n💰 Revenue Metrics:")
    print(f"   Total Revenue: ${df['revenue'].sum():.2f}")
    print(f"   Average Revenue per Session: ${df['revenue'].mean():.2f}")
    print(f"   Total Transactions: {int(df['transactions'].sum())}")
    
    print(f"\n🎯 Engagement Metrics:")
    print(f"   Average Session Duration: {df['session_duration_minutes'].mean():.2f} minutes")
    print(f"   Average Pages Viewed: {df['pages_viewed'].mean():.2f}")
    print(f"   Average Clicks: {df['clicks'].mean():.2f}")
    print(f"   Overall Bounce Rate: {df['bounce'].mean()*100:.2f}%")
    
    print(f"\n📱 Device Distribution:")
    device_dist = df['device_type'].value_counts()
    for device, count in device_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {device}: {count} sessions ({percentage:.1f}%)")
    
    print(f"\n👥 User Activity Distribution:")
    print(f"   Min Sessions per User: {int(user_stats['total_sessions'].min())}")
    print(f"   Max Sessions per User: {int(user_stats['total_sessions'].max())}")
    print(f"   Avg Sessions per User: {user_stats['total_sessions'].mean():.2f}")
    
    print("\n" + "=" * 70)


def main():
    """
    Main function to execute the complete analytics pipeline.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "BIG DATA ANALYTICS - USER ACTIVITY SESSION ANALYSIS" + " " * 7 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Step 1: Generate synthetic dataset
    df = generate_synthetic_dataset(num_users=100, num_sessions=1000, random_seed=42)
    
    # Step 2: Filter and aggregate sessions
    user_stats = filter_and_aggregate_sessions(df)
    
    # Step 3: Compute performance metric
    user_stats = compute_performance_metric(user_stats)
    
    # Step 4: Identify top users
    top_users = identify_top_users(user_stats, top_n=10)
    
    # Step 5: Display summary statistics
    display_summary_statistics(df, user_stats)
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "ANALYSIS COMPLETED SUCCESSFULLY!" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    return df, user_stats, top_users


if __name__ == '__main__':
    df, user_stats, top_users = main()
