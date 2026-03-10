import concurrent.futures
import time
import random

equipment_list = [
    {"id": 1, "name": "Soil Sensor #1", "size": 2},
    {"id": 2, "name": "Soil Sensor #2", "size": 4},
    {"id": 3, "name": "Irrigation Pump #1", "size": 5},
    {"id": 4, "name": "Irrigation Pump #2", "size": 3},
    {"id": 5, "name": "Tractor", "size": 8},
    {"id": 6, "name": "Sprinkler #1", "size": 2},
    {"id": 7, "name": "Sprinkler #2", "size": 4},
    {"id": 8, "name": "Harvesting Machine #1", "size": 6},
    {"id": 9, "name": "Harvesting Machine #2", "size": 7},
    {"id": 10, "name": "Soil Sensor #3", "size": 3},
]


def check_equipment(equipment_size):
    """
    Simulate maintenance check for agricultural equipment.
    
    Real-life scenario: Checking equipment health based on its size, where larger equipment requires more time.
    
    Args:
        equipment_size: Size of the equipment (affects maintenance time)
    
    Returns:
        Number of checks performed (simulated)
    """
    checks_performed = 0
    for i in range(0, 100000 * equipment_size):
        checks_performed += 1
    return checks_performed


def process_equipment(equipment):
    """
    Process a single equipment for maintenance.
    
    Real-life scenario: Equipment maintenance includes checks and calibrations.
    
    Args:
        equipment: Dictionary containing equipment details
    """
    checks = check_equipment(equipment["size"])
    print(f'🔧 Processed: {equipment["name"]} ({equipment["size"]} units) - {checks:,} checks performed')
    return checks


if __name__ == '__main__':
    print("=" * 70)
    print("AGRICULTURAL EQUIPMENT MANAGEMENT SYSTEM")
    print("Sequential vs Thread Pool vs Process Pool Execution")
    print("=" * 70)
    print(f"\n Processing {len(equipment_list)} equipment...\n")
    print("-" * 70)
    print("METHOD 1: SEQUENTIAL EXECUTION")
    print("   Processing equipment one by one (like a single worker)")
    print("-" * 70)
    
    start_time = time.perf_counter()  
    
    for equipment in equipment_list:
        process_equipment(equipment)
    
    sequential_time = time.perf_counter() - start_time
    print(f'\nSequential Execution Time: {sequential_time:.2f} seconds\n')
    

    print("-" * 70)
    print("METHOD 2: THREAD POOL EXECUTION (5 workers)")
    print("Like having 5 workers simultaneously processing equipment checks")
    print("-" * 70)
    
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        futures = [executor.submit(process_equipment, equipment) for equipment in equipment_list]
        concurrent.futures.wait(futures)
    
    thread_pool_time = time.perf_counter() - start_time
    print(f'\n Thread Pool Execution Time: {thread_pool_time:.2f} seconds\n')

    print("-" * 70)
    print("METHOD 3: PROCESS POOL EXECUTION (5 workers)")
    print("Like having 5 separate computers processing equipment checks")
    print("-" * 70)
    
    start_time = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
       
        futures = [executor.submit(process_equipment, equipment) for equipment in equipment_list]
     
        concurrent.futures.wait(futures)
    
    process_pool_time = time.perf_counter() - start_time
    print(f'\n Process Pool Execution Time: {process_pool_time:.2f} seconds\n')
    
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Sequential:    {sequential_time:.2f} seconds")
    print(f"Thread Pool:   {thread_pool_time:.2f} seconds")
    print(f"Process Pool:  {process_pool_time:.2f} seconds")
    print("-" * 70)
    
    thread_speedup = sequential_time / thread_pool_time if thread_pool_time > 0 else 0
    process_speedup = sequential_time / process_pool_time if process_pool_time > 0 else 0
    
    print(f"Thread Pool Speedup:  {thread_speedup:.2f}x faster")
    print(f"Process Pool Speedup: {process_speedup:.2f}x faster")
    print("=" * 70)
