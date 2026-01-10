"""
Big Data Analytics with PySpark - User Activity Session Analysis

This module demonstrates distributed processing using PySpark:
1. Synthetic dataset generation for user activity sessions
2. Distributed data filtering and aggregation using PySpark
3. Session-level statistics computation with Spark SQL
4. Top users identification based on performance metrics

Libraries used: PySpark, Pandas, NumPy
"""

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    print("⚠️  PySpark is not installed. Install with: pip install pyspark")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_dataset_pyspark(num_users=100, num_sessions=1000, random_seed=42):
    """
    Generate synthetic user activity session data and convert to Spark DataFrame.
    
    Args:
        num_users: Number of unique users
        num_sessions: Total number of sessions to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Spark DataFrame: Synthetic dataset with user activity sessions
    """
    if not PYSPARK_AVAILABLE:
        print("❌ PySpark is required for this function")
        return None
    
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("  PYSPARK DISTRIBUTED PROCESSING - USER ACTIVITY SESSIONS")
    print("=" * 70)
    print(f"\n📊 Generating dataset with:")
    print(f"   - {num_users} unique users")
    print(f"   - {num_sessions} total sessions")
    print()
    
    # Generate user IDs
    user_ids = np.random.randint(1, num_users + 1, num_sessions)
    
    # Generate session data
    session_data = {
        'user_id': user_ids.tolist(),
        'session_id': [f'SESSION_{i:05d}' for i in range(1, num_sessions + 1)],
        'session_duration_minutes': np.random.exponential(scale=30, size=num_sessions).tolist(),
        'pages_viewed': np.random.poisson(lam=10, size=num_sessions).tolist(),
        'clicks': np.random.poisson(lam=15, size=num_sessions).tolist(),
        'transactions': np.random.binomial(n=1, p=0.15, size=num_sessions).tolist(),
        'revenue': np.random.lognormal(mean=3.5, sigma=1.0, size=num_sessions).tolist(),
        'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], 
                                       size=num_sessions, 
                                       p=[0.5, 0.4, 0.1]).tolist(),
        'bounce': np.random.binomial(n=1, p=0.3, size=num_sessions).tolist()
    }
    
    # Generate timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [str(start_date + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    )) for _ in range(num_sessions)]
    
    session_data['timestamp'] = timestamps
    
    # Create Pandas DataFrame first
    df_pandas = pd.DataFrame(session_data)
    
    # Adjust revenue for transactions
    df_pandas.loc[df_pandas['transactions'] == 0, 'revenue'] = 0
    
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("UserActivityAnalytics") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")
    
    # Convert to Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas)
    
    print(f"✅ Dataset generated successfully!")
    print(f"   - Spark DataFrame created with {df_spark.count()} rows")
    print(f"\n📋 Dataset schema:")
    print("-" * 70)
    df_spark.printSchema()
    print("-" * 70)
    
    print(f"\n📋 Dataset preview (first 5 rows):")
    print("-" * 70)
    df_spark.show(5, truncate=False)
    print("-" * 70)
    
    return df_spark, spark


def filter_and_aggregate_pyspark(df_spark):
    """
    Filter and aggregate session data using PySpark operations.
    
    Args:
        df_spark: Spark DataFrame with session data
        
    Returns:
        Spark DataFrame: Aggregated statistics per user
    """
    print("\n" + "=" * 70)
    print("  PYSPARK DATA FILTERING AND AGGREGATION")
    print("=" * 70)
    
    # Filter out bounced sessions
    engaged_sessions = df_spark.filter(F.col('bounce') == 0)
    
    print(f"\n🔍 Filtering bounced sessions (distributed processing):")
    print(f"   - Total sessions: {df_spark.count()}")
    print(f"   - Bounced sessions: {df_spark.filter(F.col('bounce') == 1).count()}")
    print(f"   - Engaged sessions: {engaged_sessions.count()}")
    
    # Compute user-level aggregations using PySpark
    print(f"\n📊 Computing session-level statistics per user (distributed)...")
    
    user_stats = df_spark.groupBy('user_id').agg(
        F.count('session_id').alias('total_sessions'),
        F.mean('session_duration_minutes').alias('avg_session_duration'),
        F.sum('session_duration_minutes').alias('total_session_duration'),
        F.mean('pages_viewed').alias('avg_pages_viewed'),
        F.sum('pages_viewed').alias('total_pages_viewed'),
        F.mean('clicks').alias('avg_clicks'),
        F.sum('clicks').alias('total_clicks'),
        F.sum('transactions').alias('total_transactions'),
        F.sum('revenue').alias('total_revenue'),
        F.mean('bounce').alias('bounce_rate')
    )
    
    # Compute additional metrics
    user_stats = user_stats.withColumn(
        'engagement_score',
        (F.col('avg_pages_viewed') * 0.3 +
         F.col('avg_clicks') * 0.3 +
         F.col('avg_session_duration') * 0.2 +
         (1 - F.col('bounce_rate')) * 50 * 0.2)
    )
    
    user_stats = user_stats.withColumn(
        'conversion_rate',
        (F.col('total_transactions') / F.col('total_sessions') * 100)
    )
    
    user_stats = user_stats.withColumn(
        'avg_revenue_per_session',
        (F.col('total_revenue') / F.col('total_sessions'))
    )
    
    print(f"\n✅ Aggregation completed!")
    print(f"   - Total unique users: {user_stats.count()}")
    print(f"\n📋 Aggregated statistics preview (first 5 users):")
    print("-" * 70)
    user_stats.show(5, truncate=False)
    print("-" * 70)
    
    return user_stats


def compute_performance_metric_pyspark(user_stats):
    """
    Compute a comprehensive performance metric using Spark SQL operations.
    
    Args:
        user_stats: Spark DataFrame with user-level statistics
        
    Returns:
        Spark DataFrame: User statistics with performance score
    """
    print("\n" + "=" * 70)
    print("  PYSPARK PERFORMANCE METRIC CALCULATION")
    print("=" * 70)
    
    print("\n🎯 Computing comprehensive performance score (distributed)...")
    print("   Performance metric combines:")
    print("   - Engagement score (30%)")
    print("   - Total revenue (30%)")
    print("   - Conversion rate (20%)")
    print("   - Total sessions (20%)")
    
    # Compute min/max for normalization
    stats = user_stats.agg(
        F.min('engagement_score').alias('eng_min'),
        F.max('engagement_score').alias('eng_max'),
        F.min('total_revenue').alias('rev_min'),
        F.max('total_revenue').alias('rev_max'),
        F.min('conversion_rate').alias('conv_min'),
        F.max('conversion_rate').alias('conv_max'),
        F.min('total_sessions').alias('sess_min'),
        F.max('total_sessions').alias('sess_max')
    ).collect()[0]
    
    # Normalize and compute weighted score
    user_stats = user_stats.withColumn(
        'norm_engagement',
        (F.col('engagement_score') - stats['eng_min']) / 
        (stats['eng_max'] - stats['eng_min'])
    )
    
    user_stats = user_stats.withColumn(
        'norm_revenue',
        (F.col('total_revenue') - stats['rev_min']) / 
        (stats['rev_max'] - stats['rev_min'])
    )
    
    user_stats = user_stats.withColumn(
        'norm_conversion',
        (F.col('conversion_rate') - stats['conv_min']) / 
        (stats['conv_max'] - stats['conv_min'])
    )
    
    user_stats = user_stats.withColumn(
        'norm_sessions',
        (F.col('total_sessions') - stats['sess_min']) / 
        (stats['sess_max'] - stats['sess_min'])
    )
    
    # Compute weighted performance score
    user_stats = user_stats.withColumn(
        'performance_score',
        (F.col('norm_engagement') * 0.30 +
         F.col('norm_revenue') * 0.30 +
         F.col('norm_conversion') * 0.20 +
         F.col('norm_sessions') * 0.20) * 100
    )
    
    # Drop intermediate columns
    user_stats = user_stats.drop('norm_engagement', 'norm_revenue', 
                                  'norm_conversion', 'norm_sessions')
    
    print("\n✅ Performance scores calculated successfully!")
    
    return user_stats


def identify_top_users_pyspark(user_stats, top_n=10):
    """
    Identify and display top users using PySpark operations.
    
    Args:
        user_stats: Spark DataFrame with user statistics and performance scores
        top_n: Number of top users to display
        
    Returns:
        Spark DataFrame: Top users by performance score
    """
    print("\n" + "=" * 70)
    print(f"  TOP {top_n} USERS BY PERFORMANCE SCORE (PYSPARK)")
    print("=" * 70)
    
    # Sort by performance score and get top N
    top_users = user_stats.orderBy(F.col('performance_score').desc()).limit(top_n)
    
    # Add rank using window function
    window_spec = Window.orderBy(F.col('performance_score').desc())
    top_users = top_users.withColumn('rank', F.row_number().over(window_spec))
    
    # Reorder columns
    cols = ['rank', 'user_id', 'performance_score', 'total_sessions',
            'engagement_score', 'total_revenue', 'conversion_rate']
    
    print("\n🏆 Top Performing Users:")
    print("-" * 70)
    top_users.select(cols).show(top_n, truncate=False)
    print("-" * 70)
    
    # Convert to Pandas for detailed display
    top_users_pd = top_users.toPandas()
    
    print("\n📊 Detailed Statistics for Top 5 Users:")
    print("-" * 70)
    
    for idx, row in top_users_pd.head(5).iterrows():
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


def display_summary_statistics_pyspark(df_spark, user_stats):
    """
    Display summary statistics using PySpark aggregations.
    
    Args:
        df_spark: Original Spark DataFrame with sessions
        user_stats: User-level statistics Spark DataFrame
    """
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY STATISTICS (PYSPARK)")
    print("=" * 70)
    
    # Get overall statistics
    total_sessions = df_spark.count()
    unique_users = df_spark.select('user_id').distinct().count()
    
    revenue_stats = df_spark.agg(
        F.sum('revenue').alias('total_revenue'),
        F.mean('revenue').alias('avg_revenue')
    ).collect()[0]
    
    engagement_stats = df_spark.agg(
        F.mean('session_duration_minutes').alias('avg_duration'),
        F.mean('pages_viewed').alias('avg_pages'),
        F.mean('clicks').alias('avg_clicks'),
        F.mean('bounce').alias('bounce_rate')
    ).collect()[0]
    
    total_transactions = df_spark.agg(F.sum('transactions')).collect()[0][0]
    
    device_dist = df_spark.groupBy('device_type').count().collect()
    
    user_activity = user_stats.agg(
        F.min('total_sessions').alias('min_sessions'),
        F.max('total_sessions').alias('max_sessions'),
        F.mean('total_sessions').alias('avg_sessions')
    ).collect()[0]
    
    # Get date range
    date_stats = df_spark.agg(
        F.min('timestamp').alias('min_date'),
        F.max('timestamp').alias('max_date')
    ).collect()[0]
    
    print(f"\n📈 Dataset Overview:")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Unique Users: {unique_users}")
    print(f"   Date Range: {str(date_stats['min_date'])[:10]} to {str(date_stats['max_date'])[:10]}")
    
    print(f"\n💰 Revenue Metrics:")
    print(f"   Total Revenue: ${revenue_stats['total_revenue']:.2f}")
    print(f"   Average Revenue per Session: ${revenue_stats['avg_revenue']:.2f}")
    print(f"   Total Transactions: {int(total_transactions)}")
    
    print(f"\n🎯 Engagement Metrics:")
    print(f"   Average Session Duration: {engagement_stats['avg_duration']:.2f} minutes")
    print(f"   Average Pages Viewed: {engagement_stats['avg_pages']:.2f}")
    print(f"   Average Clicks: {engagement_stats['avg_clicks']:.2f}")
    print(f"   Overall Bounce Rate: {engagement_stats['bounce_rate']*100:.2f}%")
    
    print(f"\n📱 Device Distribution:")
    for row in device_dist:
        percentage = (row['count'] / total_sessions) * 100
        print(f"   {row['device_type']}: {row['count']} sessions ({percentage:.1f}%)")
    
    print(f"\n👥 User Activity Distribution:")
    print(f"   Min Sessions per User: {int(user_activity['min_sessions'])}")
    print(f"   Max Sessions per User: {int(user_activity['max_sessions'])}")
    print(f"   Avg Sessions per User: {user_activity['avg_sessions']:.2f}")
    
    print("\n" + "=" * 70)


def main():
    """
    Main function to execute the complete PySpark analytics pipeline.
    """
    if not PYSPARK_AVAILABLE:
        print("\n❌ PySpark is not installed!")
        print("Install with: pip install pyspark")
        print("\nAlternatively, use the standard version: user_activity_analytics.py")
        return
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 5 + "BIG DATA ANALYTICS WITH PYSPARK - USER ACTIVITY ANALYSIS" + " " * 6 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Step 1: Generate synthetic dataset
    result = generate_synthetic_dataset_pyspark(num_users=100, num_sessions=1000, random_seed=42)
    if result is None:
        return
    df_spark, spark = result
    
    # Step 2: Filter and aggregate sessions
    user_stats = filter_and_aggregate_pyspark(df_spark)
    
    # Step 3: Compute performance metric
    user_stats = compute_performance_metric_pyspark(user_stats)
    
    # Step 4: Identify top users
    top_users = identify_top_users_pyspark(user_stats, top_n=10)
    
    # Step 5: Display summary statistics
    display_summary_statistics_pyspark(df_spark, user_stats)
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "ANALYSIS COMPLETED SUCCESSFULLY!" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    # Stop Spark session
    spark.stop()
    
    return df_spark, user_stats, top_users


if __name__ == '__main__':
    main()
