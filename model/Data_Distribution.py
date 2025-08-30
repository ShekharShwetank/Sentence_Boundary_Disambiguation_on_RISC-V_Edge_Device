# understand the dataset distribution and print some samples
import numpy as np

# count distribution
num_eos = np.sum(y == 1)
num_neos = np.sum(y == 0)
print(f"Total examples: {len(y)}")
print(f"EOS=1: {num_eos}  ({num_eos/len(y):.2%})")
print(f"NEOS=0: {num_neos}  ({num_neos/len(y):.2%})")

# print a few samples for each class
print("\n--- Samples with Label = 1 (EOS) ---")
for i in np.where(y == 1)[0][:20]:
    window = "".join(inv_char2id.get(c, "?") for c in X[i]).replace("<PAD>", " ")
    print(f"Window: '{window}'  Label: {y[i]}")

print("\n--- Samples with Label = 0 (NEOS) ---")
for i in np.where(y == 0)[0][:20]:
    window = "".join(inv_char2id.get(c, "?") for c in X[i]).replace("<PAD>", " ")
    print(f"Window: '{window}'  Label: {y[i]}")
