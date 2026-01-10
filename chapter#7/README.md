# Chapter 7: Big Data Analytics - User Activity Session Analysis

## Overview

This chapter demonstrates comprehensive Big Data Analytics techniques for analyzing user activity sessions. The implementation showcases synthetic dataset generation, data filtering and aggregation, session-level statistics computation, and top user identification based on performance metrics.

## Problem Statement

Use Python (Google Colab compatible) to complete all tasks:
- Generate a synthetic dataset representing user activity sessions
- Perform filtering and aggregation to compute session-level statistics
- Identify and display the top users based on a derived performance metric
- Use appropriate Big Data Analytics libraries (Pandas, NumPy, Scikit-learn)

## Features

### 1. Synthetic Dataset Generation
- Generates realistic user activity session data with multiple features:
  - User IDs and Session IDs
  - Session duration (exponential distribution)
  - Pages viewed (Poisson distribution)
  - Clicks (Poisson distribution)
  - Transactions (binomial distribution)
  - Revenue (log-normal distribution)
  - Device types (Desktop, Mobile, Tablet)
  - Bounce rates
  - Timestamps spanning 30 days

### 2. Data Filtering and Aggregation
- Filters out bounced sessions for engagement analysis
- Computes user-level aggregations:
  - Total sessions per user
  - Average and total session duration
  - Average and total pages viewed
  - Average and total clicks
  - Total transactions and revenue
  - Bounce rate per user
  - Engagement score (composite metric)
  - Conversion rate
  - Average revenue per session

### 3. Performance Metric Calculation
- Comprehensive performance scoring system using StandardScaler normalization
- Weighted combination of key metrics:
  - Engagement score (30%)
  - Total revenue (30%)
  - Conversion rate (20%)
  - Total sessions (20%)
- Scaled to 0-100 range for interpretability

### 4. Top Users Identification
- Ranks users by performance score
- Displays detailed statistics for top performers
- Shows key metrics including:
  - Performance score
  - Total sessions
  - Total revenue
  - Engagement score
  - Conversion rate
  - Session duration
  - Bounce rate

## Technologies Used

- **Pandas**: Data manipulation and aggregation
- **NumPy**: Numerical computing and random data generation
- **Scikit-learn**: Machine learning preprocessing (StandardScaler)
- **PySpark** (optional): Distributed data processing
- **Python datetime**: Timestamp generation and manipulation

## Files Included

1. **user_activity_analytics.py** - Standard implementation using Pandas, NumPy, and Scikit-learn
2. **user_activity_analytics_pyspark.py** - Distributed processing version using PySpark
3. **user_activity_analytics_colab.ipynb** - Google Colab notebook (optional)

## Installation

### For Standard Version (Pandas/NumPy/Scikit-learn)

Install required libraries:

```bash
pip install pandas numpy scikit-learn
```

### For PySpark Version

Install PySpark additionally:

```bash
pip install pandas numpy scikit-learn pyspark
```

## Usage

### Standard Version (Pandas/NumPy/Scikit-learn)

Run the script directly:

```bash
python user_activity_analytics.py
```

### PySpark Version (Distributed Processing)

Run the PySpark script:

```bash
python user_activity_analytics_pyspark.py
```

Note: PySpark version provides distributed processing capabilities, ideal for larger datasets.

### Programmatic Usage

Or import and use individual functions:

```python
from user_activity_analytics import (
    generate_synthetic_dataset,
    filter_and_aggregate_sessions,
    compute_performance_metric,
    identify_top_users
)

# Generate dataset
df = generate_synthetic_dataset(num_users=100, num_sessions=1000)

# Process data
user_stats = filter_and_aggregate_sessions(df)
user_stats = compute_performance_metric(user_stats)

# Get top users
top_users = identify_top_users(user_stats, top_n=10)
```

## Sample Output

```
╔════════════════════════════════════════════════════════════════════╗
║          BIG DATA ANALYTICS - USER ACTIVITY SESSION ANALYSIS       ║
╚════════════════════════════════════════════════════════════════════╝

📊 Generating dataset with:
   - 100 unique users
   - 1000 total sessions

✅ Dataset generated successfully!

🔍 Filtering bounced sessions:
   - Total sessions: 1000
   - Bounced sessions: 309
   - Engaged sessions: 691

📊 Computing session-level statistics per user...

🎯 Computing comprehensive performance score...
   Performance metric combines:
   - Engagement score (30%)
   - Total revenue (30%)
   - Conversion rate (20%)
   - Total sessions (20%)

🏆 Top Performing Users:
----------------------------------------------------------------------
 rank  user_id  performance_score  total_sessions  engagement_score  total_revenue  conversion_rate
    1       30         100.000000              10         18.140108     413.358918        50.000000
    2       39          99.894161              12         24.669880     308.630190        25.000000
    3       24          97.470434              14         23.006523     323.454091        21.428571
    ...

📈 Dataset Overview:
   Total Sessions: 1000
   Unique Users: 100
   Date Range: 2025-12-11 to 2026-01-10

💰 Revenue Metrics:
   Total Revenue: $7794.08
   Average Revenue per Session: $7.79
   Total Transactions: 149

╔════════════════════════════════════════════════════════════════════╗
║                    ANALYSIS COMPLETED SUCCESSFULLY!                 ║
╚════════════════════════════════════════════════════════════════════╝
```

## Key Concepts Demonstrated

### 1. Statistical Distributions
- **Exponential Distribution**: Session duration modeling (realistic user behavior)
- **Poisson Distribution**: Page views and clicks (event counting)
- **Binomial Distribution**: Transaction modeling (binary outcomes)
- **Log-Normal Distribution**: Revenue modeling (skewed positive values)

### 2. Data Aggregation
- GroupBy operations for user-level statistics
- Multiple aggregation functions (mean, sum, count)
- Derived metrics calculation

### 3. Normalization and Scaling
- StandardScaler for feature normalization
- Weighted scoring system
- Min-max scaling for interpretability

### 4. Business Intelligence
- Engagement scoring
- Conversion rate analysis
- Performance benchmarking
- User segmentation

## Real-World Applications

This implementation can be adapted for:
- E-commerce platforms analyzing user behavior
- SaaS applications tracking user engagement
- Content platforms measuring user activity
- Marketing analytics identifying high-value customers
- Product analytics understanding user patterns

## Customization

You can customize the script by modifying:

1. **Dataset size**: Change `num_users` and `num_sessions` parameters
2. **Distributions**: Adjust distribution parameters (scale, lam, p, mean, sigma)
3. **Performance weights**: Modify weights in `compute_performance_metric()`
4. **Additional features**: Add new columns to the synthetic dataset
5. **Aggregation metrics**: Add custom aggregations in `filter_and_aggregate_sessions()`

## Google Colab Compatibility

Both scripts are fully compatible with Google Colab. 

### For Standard Version:

1. Create a new Colab notebook
2. Install dependencies:
   ```python
   !pip install pandas numpy scikit-learn
   ```
3. Upload `user_activity_analytics.py` or copy the code
4. Run: `%run user_activity_analytics.py` or execute the cells

### For PySpark Version:

1. Create a new Colab notebook
2. Install dependencies:
   ```python
   !pip install pyspark
   ```
3. Upload `user_activity_analytics_pyspark.py` or copy the code
4. Run: `%run user_activity_analytics_pyspark.py` or execute the cells

Note: Google Colab provides a pre-configured environment ideal for Big Data Analytics tasks.

## Notes

- The synthetic data is generated with a fixed random seed (42) for reproducibility
- Performance scores are relative within the dataset
- The engagement score is a composite metric combining multiple factors
- Bounce rate filtering helps focus on engaged users
- All monetary values are in USD

## Future Enhancements

Potential improvements:
- Add time-series analysis for trend detection
- Implement cohort analysis
- Add visualization with matplotlib/seaborn
- Include PySpark for distributed processing
- Add NetworkX for user relationship analysis
- Implement machine learning models for prediction
- Add A/B testing framework
- Include anomaly detection

## License

This implementation is part of the Parallel and Distributed Computing course materials.
