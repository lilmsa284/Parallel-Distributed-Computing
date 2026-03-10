import asyncio
import random


async def check_security_system(end_time, loop):
    """
    Task A: Security System Check
    """
    print("[SECURITY] Checking all door and window sensors...")

    check_time = random.randint(1, 3)
    await asyncio.sleep(check_time)
    print(f"   ✓ Security check completed in {check_time}s - All sensors OK!")
    
    if (loop.time() + 1.0) < end_time:
        print("   → Scheduling: Temperature Control in 1 second...\n")
        loop.call_later(1, asyncio.create_task, adjust_temperature(end_time, loop))
    else:
        print("\nAutomation cycle complete. Shutting down...")
        loop.stop()


async def adjust_temperature(end_time, loop):
    """
    Task B: Temperature Control
    """
    current_temp = random.randint(18, 28)
    target_temp = 22
    
    print(f"[THERMOSTAT] Current temperature: {current_temp}°C")
    
    adjust_time = random.randint(1, 3)
    await asyncio.sleep(adjust_time)
    
    if current_temp < target_temp:
        print(f"   ✓ Heating activated! Adjusting to {target_temp}°C (took {adjust_time}s)")
    elif current_temp > target_temp:
        print(f"   ✓ Cooling activated! Adjusting to {target_temp}°C (took {adjust_time}s)")
    else:
        print(f"   ✓ Temperature is perfect at {target_temp}°C!")
   
    if (loop.time() + 1.0) < end_time:
        print("   → Scheduling: Lighting Control in 1 second...\n")
        loop.call_later(1, asyncio.create_task, control_lights(end_time, loop))
    else:
        print("\nAutomation cycle complete. Shutting down...")
        loop.stop()


async def control_lights(end_time, loop):
    """
    Task C: Smart Lighting Control
    """
    rooms = ["Living Room", "Bedroom", "Kitchen", "Bathroom"]
    selected_room = random.choice(rooms)
    brightness = random.randint(0, 100)
    
    print(f"[LIGHTS] Checking {selected_room} lighting...")
    
    adjust_time = random.randint(1, 3)
    await asyncio.sleep(adjust_time)
    
    print(f"   ✓ {selected_room} brightness set to {brightness}% (took {adjust_time}s)")
    
    if (loop.time() + 1.0) < end_time:
        print("   → Scheduling: Security Check in 1 second...\n")
        print("-" * 50)
        loop.call_later(1, asyncio.create_task, check_security_system(end_time, loop))
    else:
        print("\nAutomation cycle complete. Shutting down...")
        loop.stop()


if __name__ == '__main__':
    print("=" * 60)
    print("    SMART HOME AUTOMATION SYSTEM")
    print("    Using Asyncio Event Loop Scheduling")
    print("=" * 60)
    print("\nThis system rotates between three smart home tasks:")
    print("  1. Security System Check")
    print("  2. Temperature Control")
    print("  3. Lighting Control")
    print("\nRunning automation cycle for 30 seconds...")
    print("=" * 60 + "\n")
    
    loop = asyncio.get_event_loop()

    end_loop = loop.time() + 30

    # Start the first task
    loop.call_soon(asyncio.create_task, check_security_system(end_loop, loop))
    
    loop.run_forever()
   
    loop.close()
    
    print("\n" + "=" * 60)
    print("    SMART HOME AUTOMATION STOPPED")
    print("    All systems in standby mode")
    print("=" * 60)
