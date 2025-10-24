import multiprocessing
import time
import os

def worker(shared_data):
    for _ in range(5):
        shared_data.number += 3
        print(f"PID {os.getpid()} updated number to {shared_data.number}")
        time.sleep(0.5)

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_data = manager.Namespace()
    shared_data.number = 0

    process = multiprocessing.Process(target=worker, args=(shared_data,))
    process.start()
    process.join()

    print(f"Final value in namespace: {shared_data.number}")
