import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

result_dir = Path("result")
result_files = list(result_dir.glob("*.txt"))
result_files = sorted(result_files, key=lambda x: int( x.stem.split("_")[1] ))

training_data = Path("weathernews.txt").read_text().split("\n")
training_data = [l.split(" ") for l in training_data if l]
for tline in training_data:
    tline.insert(0, "<BOS>")
    tline.append("<EOS>")

iter = []
counts = []
excluded_sentences = []

for file in result_files:
    generated_data = file.read_text().split("\n")
    generated_data = [l.split(" ") for l in generated_data if l]

    cnt = 0
    excluded = []
    for gdata in generated_data:
        if gdata in training_data:
            cnt += 1
        else:
            excluded.append(" ".join(gdata))
    iter.append( int( file.stem.split("_")[1] ) )
    counts.append( cnt )
    excluded_sentences.append(excluded)

plt.plot(iter, counts)
plt.ylim(0, 100)
plt.xlabel("Iteration")
plt.ylabel("Num of generated data included in training data")

# plt.show()
# plt.savefig("count_result.png")
print(iter)
print(counts)
