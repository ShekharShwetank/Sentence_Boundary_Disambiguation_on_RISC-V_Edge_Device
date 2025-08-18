import numpy as np

X = np.load(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\X.npy")
y = np.load(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\y.npy")

print("Original distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Label {u}: {c}")

# Separate indices
idx_eos = np.where(y == 1)[0]
idx_neos = np.where(y == 0)[0]

n_eos = len(idx_eos)
n_neos = len(idx_neos)

def balance_dataset(X, y, eos_idx, neos_idx, mode="strict", ratio=5):
    if mode == "strict":
        # 1:1 balance
        target_size = min(len(eos_idx), len(neos_idx))
        sel_eos = np.random.choice(eos_idx, target_size, replace=False)
        sel_neos = np.random.choice(neos_idx, target_size, replace=True)
    elif mode == "ratio":
        # keep all NEOS, sample EOS with desired ratio
        target_size_neos = len(neos_idx)
        target_size_eos = min(len(eos_idx), ratio * target_size_neos)
        sel_neos = neos_idx  # all NEOS
        sel_eos = np.random.choice(eos_idx, target_size_eos, replace=False)
    else:
        raise ValueError("mode must be 'strict' or 'ratio'")

    sel_idx = np.concatenate([sel_eos, sel_neos])
    np.random.shuffle(sel_idx)
    return X[sel_idx], y[sel_idx]

X_strict, y_strict = balance_dataset(X, y, idx_eos, idx_neos, mode="strict")
np.save(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\X_bal_strict.npy", X_strict)
np.save(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\y_bal_strict.npy", y_strict)

print("\nStrict balance distribution:")
unique, counts = np.unique(y_strict, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Label {u}: {c}")

X_ratio, y_ratio = balance_dataset(X, y, idx_eos, idx_neos, mode="ratio", ratio=5) #1:5 in this case, adjust accordingly
np.save(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\X_bal_ratio.npy", X_ratio)
np.save(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\y_bal_ratio.npy", y_ratio)

print("\nRatio balance distribution (1:5):")
unique, counts = np.unique(y_ratio, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Label {u}: {c}")
