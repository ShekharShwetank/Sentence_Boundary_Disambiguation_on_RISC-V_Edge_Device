import re
import numpy as np
import random
import string
from pathlib import Path
import nltk

# ensure punkt tokenizer is available
nltk.download("punkt", quiet=True)

WINDOW_RADIUS = 10
WINDOW_LEN = 2 * WINDOW_RADIUS + 1
PUNCTUATION = {".", "?", "!"}
MAX_EXAMPLES = 100_000

alphabet = string.ascii_lowercase + string.digits + string.punctuation + " "
char2id = {c: i + 1 for i, c in enumerate(alphabet)}
char2id["<PAD>"] = 0

def encode_window(window: str):
    return [char2id.get(ch.lower(), 0) for ch in window]

def clean_text(line: str) -> str:
    return re.sub(r"<.*?>", "", line).strip()

def extract_windows_from_paragraph(paragraph, X, y, max_examples):
    # split paragraph into sentences
    sentences = nltk.sent_tokenize(paragraph)
    for sent in sentences:
        for idx, ch in enumerate(sent):
            if ch in PUNCTUATION:
                start = max(0, idx - WINDOW_RADIUS)
                end = idx + WINDOW_RADIUS + 1
                window = sent[start:end]
                if len(window) < WINDOW_LEN:
                    window = window.ljust(WINDOW_LEN, " ")
                # EOS if this punctuation is last char of the sentence
                label = 1 if idx == len(sent) - 1 else 0
                X.append(encode_window(window))
                y.append(label)
                if len(X) >= max_examples:
                    return

def build_dataset(corpus_dir: Path, max_examples=MAX_EXAMPLES):
    X, y = [], []
    files = sorted(corpus_dir.glob("*.txt"))
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            buffer = []
            for raw_line in f:
                line = clean_text(raw_line)
                if line == "":
                    if buffer:
                        paragraph = " ".join(buffer)
                        extract_windows_from_paragraph(paragraph, X, y, max_examples)
                        buffer = []
                        if len(X) >= max_examples:
                            return np.array(X, dtype=np.int32), np.array(y, dtype=np.int8)
                else:
                    buffer.append(line)
            if buffer:
                paragraph = " ".join(buffer)
                extract_windows_from_paragraph(paragraph, X, y, max_examples)
                if len(X) >= max_examples:
                    break
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int8)

if __name__ == "__main__":
    corpus_dir = Path(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\txt\en")

    X, y = build_dataset(corpus_dir, max_examples=MAX_EXAMPLES)

    np.save(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\X.npy", X)
    np.save(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\y.npy", y)

    print("Dataset built:", X.shape, y.shape)

    idxs = random.sample(range(len(X)), min(100000, len(X)))
    inv_char2id = {v: k for k, v in char2id.items()}

    for i in idxs:
        window = "".join(inv_char2id.get(c, "?") for c in X[i]).replace("<PAD>", " ")
        print(f"Window: '{window}'  Label: {y[i]}")


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
