
import random 

with open(f"../func/not_cbench_cg.txt", "r") as benchmarks_files:
  benchmarks = benchmarks_files.readlines()


for i in range(0,10):
    test_set = random.choices(benchmarks,k=5)
    print(test_set)
