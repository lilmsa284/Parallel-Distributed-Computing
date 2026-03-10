import asyncio


@asyncio.coroutine
def calculate_daily_sales(num_orders):
    """
    Analytics Task 1: Calculate Daily Sales Revenue
    
    Real-life scenario: E-commerce platform calculates total daily revenue
    by processing each order one by one.
    
    Args:
        num_orders: Number of orders to process
    """
    total_revenue = 0
    base_price = 25  # Base price per order ($25)
    
    for i in range(1, num_orders + 1):
        order_value = base_price * i  # Each order has increasing value
        total_revenue += order_value
        print(f'📊 [SALES] Processing order #{i} - Value: ${order_value}')
        yield from asyncio.sleep(1)
    
    print(f'\n✅ [SALES COMPLETE] Total Daily Revenue = ${total_revenue}')
    print(f'   Average Order Value = ${total_revenue / num_orders:.2f}\n')


@asyncio.coroutine
def track_customer_visits(num_hours):
    """
    Analytics Task 2: Track Customer Visit Patterns
    
    Real-life scenario: Website analytics tracking visitor patterns
    using Fibonacci-like growth pattern for realistic traffic simulation.
    
    Args:
        num_hours: Number of hours to analyze
    """
    previous_visitors, current_visitors = 10, 15  # Starting visitors
    total_visitors = previous_visitors
    
    for hour in range(num_hours):
        print(f'👥 [TRAFFIC] Hour {hour + 1}: {current_visitors} visitors on site')
        yield from asyncio.sleep(1)
        total_visitors += current_visitors
        previous_visitors, current_visitors = current_visitors, previous_visitors + current_visitors
    
    print(f'\n✅ [TRAFFIC COMPLETE] Total Visitors Today = {total_visitors}')
    print(f'   Peak Hour Visitors = {current_visitors}\n')


@asyncio.coroutine
def analyze_inventory_turnover(total_products, sold_products):
    """
    Analytics Task 3: Analyze Inventory Turnover Rate
    
    Real-life scenario: Warehouse management system calculates
    inventory turnover ratio to optimize stock levels.
    
    Args:
        total_products: Total products in inventory
        sold_products: Number of products sold
    """
    turnover_rate = 1
    
    for i in range(1, sold_products + 1):
        # Calculate progressive turnover using combination formula
        turnover_rate = turnover_rate * (total_products - i + 1) / i
        print(f'📦 [INVENTORY] Analyzing product batch #{i} - Turnover factor: {turnover_rate:.0f}')
        yield from asyncio.sleep(1)
    
    efficiency = min(100, (sold_products / total_products) * 100)
    print(f'\n✅ [INVENTORY COMPLETE] Turnover Combinations = {turnover_rate:.0f}')
    print(f'   Stock Efficiency = {efficiency:.1f}%\n')


if __name__ == '__main__':
    print("=" * 65)
    print("    E-COMMERCE ORDER ANALYTICS SYSTEM")
    print("    Using Asyncio.Task for Parallel Execution")
    print("=" * 65)
    print("\nRunning three analytics tasks in PARALLEL:")
    print("  📊 Task 1: Daily Sales Revenue Calculation")
    print("  👥 Task 2: Customer Traffic Analysis")
    print("  📦 Task 3: Inventory Turnover Analysis")
    print("\n" + "-" * 65 + "\n")
    
    # Create task list - all three tasks run concurrently
    task_list = [
        asyncio.Task(calculate_daily_sales(8)),           # Process 8 orders
        asyncio.Task(track_customer_visits(8)),           # Analyze 8 hours
        asyncio.Task(analyze_inventory_turnover(15, 8))   # 15 products, 8 sold
    ]
    
    # Get event loop and run all tasks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(task_list))
    loop.close()
    
    print("-" * 65)
    print("\n" + "=" * 65)
    print("    ALL ANALYTICS TASKS COMPLETED!")
    print("    Dashboard ready for business review")
    print("=" * 65)