import multiprocessing
import time
import os

def task():
    print(f"Process {os.getpid()} started a task")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    process = multiprocessing.Process(target=task)
    process.start()
    print(f"Started process with PID: {process.pid}")

    time.sleep(3)  

    print("Terminating the process...")
    process.terminate()
    process.join()

    print("Process terminated successfully")
