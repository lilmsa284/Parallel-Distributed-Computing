import asyncio
import random

async def check_soil_moisture(end_time, loop):
    """
    Task A: Soil Moisture Check
    
    Real-life scenario: Checking soil moisture to determine if crops need watering
    - Reads moisture levels
    - Activates irrigation if necessary
    
    Args:
        end_time: Time when the automation cycle should end
        loop: The asyncio event loop
    """
    print("[CROPS] Checking soil moisture levels...")

    check_time = random.randint(1, 3)
    await asyncio.sleep(check_time)
    moisture_level = random.randint(0, 100)
    
    if moisture_level < 30:
        print(f"Moisture level low ({moisture_level}%) - Activating irrigation!")
    else:
        print(f"Moisture level sufficient ({moisture_level}%)")
    
    if (loop.time() + 1.0) < end_time:
        print("   → Scheduling: Livestock Feeding in 1 second...\n")
        # Schedule next task as a coroutine with asyncio.create_task()
        asyncio.create_task(feed_livestock(end_time, loop))
    else:
        print("\n⏰ Automation cycle complete. Shutting down...")


async def feed_livestock(end_time, loop):
    """
    Task B: Livestock Feeding
    
    Real-life scenario: Feeding livestock at scheduled intervals
    - Checks scheduled feeding times
    - Distributes food accordingly
    
    Args:
        end_time: Time when the automation cycle should end
        loop: The asyncio event loop
    """
    animals = ["Cows", "Sheep", "Chickens", "Goats"]
    selected_animal = random.choice(animals)
    food_amount = random.randint(1, 10)  
    
    print(f" [FEEDING] Feeding {selected_animal} with {food_amount}kg of food...")

    feeding_time = random.randint(1, 3)
    await asyncio.sleep(feeding_time)
    
    print(f" {selected_animal} fed (took {feeding_time}s)")

    if (loop.time() + 1.0) < end_time:
        print("Scheduling: Temperature Monitoring in 1 second...\n")
        asyncio.create_task(monitor_temperature(end_time, loop))
    else:
        print("\nAutomation cycle complete. Shutting down...")


async def monitor_temperature(end_time, loop):
    """
    Task C: Temperature Monitoring
    
    Real-life scenario: Monitoring temperature of animal shelters
    - Checks current temperature
    - Adjusts heating or cooling based on thresholds
    
    Args:
        end_time: Time when the automation cycle should end
        loop: The asyncio event loop
    """
    shelter = random.choice(["Cow Shed", "Sheep Barn", "Chicken Coop", "Goat Pen"])
    current_temp = random.randint(10, 30)  
    
    print(f"Checking temperature in {shelter}...")

    check_time = random.randint(1, 3)
    await asyncio.sleep(check_time)

    if current_temp < 18:
        print(f"Temperature low ({current_temp}°C) - Activating heating system!")
    elif current_temp > 25:
        print(f"Temperature high ({current_temp}°C) - Activating cooling system!")
    else:
        print(f"Temperature is optimal at {current_temp}°C.")
    
    if (loop.time() + 1.0) < end_time:
        print("   → Scheduling: Soil Moisture Check in 1 second...\n")
        print("-" * 50)
        asyncio.create_task(check_soil_moisture(end_time, loop))
    else:
        print("\nAutomation cycle complete. Shutting down...")


async def main():
    """
    Main function to run the smart farm tasks for 30 seconds.
    """
    print("=" * 60)
    print("SMART FARM MANAGEMENT SYSTEM")
    print("Using Asyncio Event Loop Scheduling")
    print("=" * 60)
    print("\nThis system rotates between three smart farm tasks:")
    print("1. Soil Moisture Check (Crops)")
    print("2. Livestock Feeding")
    print("3. Temperature Monitoring (Animal Shelters)")
    print("\nRunning automation cycle for 30 seconds...")
    print("=" * 60 + "\n")
    
    loop = asyncio.get_event_loop()

    end_loop = loop.time() + 30

    # Start the first task (check soil moisture)
    asyncio.create_task(check_soil_moisture(end_loop, loop))
    
    # Wait until the loop completes all tasks (tasks are chained)
    await asyncio.sleep(30)  # Keep the loop running for 30 seconds
    print("\n" + "=" * 60)
    print("SMART FARM MANAGEMENT STOPPED")
    print("All systems in standby mode")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
