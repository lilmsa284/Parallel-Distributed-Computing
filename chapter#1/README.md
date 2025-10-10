Multiprocessing vs Multithreading in Python

*Overview*

This project demonstrates the performance differences between multiprocessing and multithreading in Python. By running a computationally intensive function (do_something()), it compares the execution times of both approaches.

Because of Python’s Global Interpreter Lock (GIL), threads cannot truly run in parallel for CPU-heavy tasks, making multiprocessing generally more efficient in such cases.

*How It Works*

do_something() function

The workload is defined in a separate file (do_something.py). It simulates either a CPU-bound or I/O-bound operation.

Example (CPU-bound):

def do_something(size, out_list):
    for i in range(size):
        out_list.append(i ** 2)  # Example: squaring numbers


You can easily swap this with other tasks, such as reading files, making API calls, or performing matrix computations.

*Main script (multithreading_test.py)*

The main program imports do_something() and executes it using two methods:

Multiprocessing (e.g., 10 processes)

Multithreading (e.g., 10 threads)

The script records how long each method takes and prints a side-by-side comparison.

*Running the Program*

Place both files in the same folder.

Run the main script:

python multithreading_test.py


Compare the execution times in the output.

*Key Observations*

Multiprocessing consistently outperforms multithreading for CPU-heavy tasks.

Increasing the number of workers (processes/threads) eventually slows performance due to context switching and resource overhead.

Multithreading struggles with CPU-bound workloads because of the GIL, which limits parallel execution of threads.

Multiprocessing leverages multiple CPU cores effectively, making it better suited for computationally demanding operations.

*Conclusion*

Multiprocessing is the preferred choice for CPU-bound workloads, since each process runs independently on a separate core.

Multithreading is more effective for I/O-bound tasks (like handling network requests or disk I/O), where concurrency matters more than raw CPU speed.

In this experiment, multiprocessing ran up to 3x faster than multithreading for the same task.