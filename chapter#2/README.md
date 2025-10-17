THREAD SYNCHRONIZATION

Overview:
This program showcases thread synchronization in Python using three distinct mechanisms:

1) Lock
![alt text](image.png)

2) RLock
![alt text](image-1.png)

3) Semaphore
![alt text](image-2.png)



Each synchronization technique is evaluated with multiple threads executing simulated tasks. The total execution time for all threads is recorded to compare performance across methods.


Observations:

RLock showed slightly better performance overall compared to Lock and Semaphore. As the number of threads increased, execution time scaled approximately linearly. Semaphore offers more control over access limits but may introduce slight overhead. While Lock guarantees mutual exclusion, it cannot be re-acquired by the same thread—unlike RLock.

Conclusion:

All three synchronization methods effectively prevent race conditions. RLock offers a good balance between flexibility and ease of use. Semaphore is ideal when you need to limit the number of threads accessing a shared resource at the same time.