from parse import compile

import matplotlib.pyplot as plt

template = compile("Average time: {}ms")

def canonicalize(s):
    return float(s.strip())

def parse(lines):
    results = []
    for line in lines:
        if result := template.parse(line):
            results.append(canonicalize(result[0]))
    return results

singlePasses = []
multiPasses = []
for name in [f"./data/compare/compare_{i}.txt" for i in range(1, 10)]:
    with open(name) as f:
        singlePasses.append(parse(f.read().splitlines()))

for name in [f"./data/compare/multipass_compare_{i}.txt" for i in range(1, 10)]:
    with open(name) as f:
        multiPasses.append(parse(f.read().splitlines()))

with open('./data/pow2Reduction.txt') as f:
    atomicSinglePass = parse(f.read().splitlines())

legend = [f"{2**blocks} Blocks" for blocks in range(1, 10)]

plt.figure(1)

plt.title("Atomic Single Pass")
plt.plot(atomicSinglePass)
plt.xlabel("Data Size (log2)")
plt.ylabel("Time [ms]")

plt.figure(2)

plt.title("Singlepass Execution Times")
for sample in singlePasses:
    plt.plot(sample)
plt.xlabel("Data Size (log2)")
plt.ylabel("Time [ms]")
plt.legend(legend)

plt.figure(3)

plt.title("Multipass Execution Times")
for sample in multiPasses:
    plt.plot(sample)

plt.xlabel("Data Size (log2)")
plt.ylabel("Time [ms]")
plt.legend(legend)

plt.figure(4)


singleP = []
multiP = []

for i in range(28):
    singleP.append(max([row[i] for row in singlePasses]))
    multiP.append(max([row[i] for row in multiPasses]))

for singleP, multiP in zip(singlePasses, multiPasses):
    plt.plot([(single - multi) for single, multi in zip(singleP, multiP)])

plt.title("Difference plot (single - multi)")
plt.xlabel("Data Size (log2)")
plt.legend(legend, loc='upper left')

plt.show()

