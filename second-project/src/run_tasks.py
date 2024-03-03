import time

from task_1 import task_1
from task_2 import task_2

if __name__ == '__main__':
    print("Running tasks...")
    print("Task 1...")
    start_task_1 = time.time()
    task_1(verbose=True)
    end_task_1 = time.time()
    print(f"Task 1 took {end_task_1 - start_task_1} seconds")

    print("Task 2...")
    start_task_2 = time.time()
    task_2()
    end_task_2 = time.time()
    print(f"Task 2 took {end_task_2 - start_task_2} seconds")
    print( f"Finished both tasks after {end_task_2 - start_task_1} seconds")
