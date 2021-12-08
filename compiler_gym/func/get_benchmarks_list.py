#makes list of all known benchmarks available in compiler_gym
import compiler_gym
env = compiler_gym.make(
     "llvm-v0",
     benchmark="cbench-v1/qsort",
     observation_space="Autophase",
     reward_space="IrInstructionCountOz"
)
env.reset()                              
env.step(env.action_space.sample())
env.close()     


with open(f"list_of_benchmarks_cg.txt", "w") as fstream:
  for count, benchmark in enumerate(env.datasets.benchmark_uris()):
    if count%50000 == 0: 
        print(f"On {count}")
        print(benchmark)
    if benchmark.startswith("benchmark"):
      fstream.write(benchmark + '\n')