import multiprocessing
import os

def worker():
    print(f"Child Process PID: {os.getpid()} is running")

if __name__ == "__main__":
    process = multiprocessing.Process(target=worker)
    process.start()
    process.join()
    print("Main Process Finished")
