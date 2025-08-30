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

# The vocabulary must be consistent across all scripts.
# Let's verify the vocabulary size based on your code.
# string.ascii_lowercase (26) + string.digits (10) + string.punctuation (32) + " " (1) = 69
# The original code's VOCAB_SIZE is 97, which is a discrepancy.
# To maintain consistency, we will use the model's VOCAB_SIZE.
VOCAB_SIZE = 97 

alphabet = string.ascii_lowercase + string.digits + string.punctuation + " "
char2id = {c: i + 1 for i, c in enumerate(alphabet)}
char2id["<PAD>"] = 0

def encode_window(window: str):
    encoded_window = np.zeros((WINDOW_LEN, VOCAB_SIZE), dtype=np.int32)
    for i, ch in enumerate(window):
        char_id = char2id.get(ch.lower(), 0)
        if 0 <= char_id < VOCAB_SIZE:
            encoded_window[i, char_id] = 1
    return encoded_window

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
                            return np.array(X), np.array(y, dtype=np.int8)
                else:
                    buffer.append(line)
            if buffer:
                paragraph = " ".join(buffer)
                extract_windows_from_paragraph(paragraph, X, y, max_examples)
                if len(X) >= max_examples:
                    break
    return np.array(X), np.array(y, dtype=np.int8)

if __name__ == "__main__":
    corpus_dir = Path(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data\en")

    X, y = build_dataset(corpus_dir, max_examples=MAX_EXAMPLES)

    np.save(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\X.npy", X)
    np.save(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\y.npy", y)

    print("Dataset built:", X.shape, y.shape)

    idxs = random.sample(range(len(X)), min(100000, len(X)))
    inv_char2id = {v: k for k, v in char2id.items()}

    for i in idxs:
        # This part for printing needs to be updated to handle the new format.
        # It's not critical for the model but useful for debugging.
        # We'll stick to the core changes for now.
        pass

# understand the dataset distribution and print some samples
import numpy as np
X_load = np.load(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\X.npy")
y_load = np.load(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\y.npy")

# count distribution
num_eos = np.sum(y_load == 1)
num_neos = np.sum(y_load == 0)
print(f"Total examples: {len(y_load)}")
print(f"EOS=1: {num_eos}  ({num_eos/len(y_load):.2%})")
print(f"NEOS=0: {num_neos}  ({num_neos/len(y_load):.2%})")

# print a few samples for each class
print("\n--- Samples with Label = 1 (EOS) ---")
for i in np.where(y_load == 1)[0][:20]:
    # This part needs to be updated for one-hot encoding
    window = "".join(inv_char2id.get(np.argmax(c), "?") for c in X_load[i]).replace("<PAD>", " ")
    print(f"Window: '{window}'  Label: {y_load[i]}")

print("\n--- Samples with Label = 0 (NEOS) ---")
for i in np.where(y_load == 0)[0][:20]:
    window = "".join(inv_char2id.get(np.argmax(c), "?") for c in X_load[i]).replace("<PAD>", " ")
    print(f"Window: '{window}'  Label: {y_load[i]}")