from __future__ import annotations
import re
from pathlib import Path
from typing import List
import h5py
import torch
import tqdm
from transformers import BertModel, BertTokenizer

# ---------------------------------------------------------------------------
# 1. Parameters – edit here if necessary
# ---------------------------------------------------------------------------
DICT_PATH = Path("./models/words.txt")
OUTPUT_H5 = "bert_midlayer_dict.h5"
PATTERN = re.compile(r"^[a-z][a-z-]*[a-z]$")
LAYERS = [6, 7]
BATCH_SIZE_GPU = 2048//2
BATCH_SIZE_CPU = 32

# ---------------------------------------------------------------------------
# 2. Load dictionary words
# ---------------------------------------------------------------------------
with DICT_PATH.open(encoding="utf8") as fh:
    words: List[str] = [w for line in fh if (w := line.strip()) and PATTERN.match(w)]
print(f"Loaded {len(words):,} words from {DICT_PATH}.")

# ---------------------------------------------------------------------------
# 3. Device selection and model loading
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Running on {device}.")

print("Loading BERT (bert-base-uncased)…")
tok = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained(
    "bert-base-uncased", output_hidden_states=True
).to(device).eval()

CLS_ID, SEP_ID = tok.cls_token_id, tok.sep_token_id
DIM = bert.config.hidden_size  # 768
BATCH_SIZE = BATCH_SIZE_GPU if device.type != "cpu" else BATCH_SIZE_CPU
print(f"Batch size: {BATCH_SIZE}")

# ---------------------------------------------------------------------------
# 4. Prepare the HDF5 store
# ---------------------------------------------------------------------------
store = h5py.File(OUTPUT_H5, "w")
store.create_dataset(
    "tokens",
    data=[w.encode("utf8") for w in words],
    dtype="S32",
    compression="gzip",
)
for L in LAYERS:
    store.create_dataset(
        f"layer{L}", shape=(len(words), DIM), dtype="float32", compression="gzip"
    )

# ---------------------------------------------------------------------------
# 5. Helper: build a padded batch
# ---------------------------------------------------------------------------
PAD_ID = tok.pad_token_id

def make_batch(batch_words: List[str]):
    """Return input_ids tensor and lengths list in *device*."""
    sub_id_lists = [tok.convert_tokens_to_ids(tok.tokenize(w)) for w in batch_words]
    lengths = [len(ids) for ids in sub_id_lists]
    max_len = max(lengths)

    # +2 for [CLS] and [SEP]
    ids = torch.full((len(batch_words), max_len + 2), PAD_ID, dtype=torch.long, device=device)
    att = torch.zeros_like(ids)

    for i, sub_ids in enumerate(sub_id_lists):
        ids[i, 0] = CLS_ID
        ids[i, 1 : 1 + len(sub_ids)] = torch.tensor(sub_ids, device=device)
        ids[i, 1 + len(sub_ids)] = SEP_ID
        att[i, : 1 + len(sub_ids) + 1] = 1  # mask real tokens incl. [SEP]
    return ids, att, lengths

# ---------------------------------------------------------------------------
# 6. Main extraction loop
# ---------------------------------------------------------------------------
print("Extracting embeddings…")

with torch.no_grad():
    for start in tqdm.trange(0, len(words), BATCH_SIZE):
        bw = words[start : start + BATCH_SIZE]
        ids, att, lengths = make_batch(bw)
        print("Doing forward pass...")
        outs = bert(input_ids=ids, attention_mask=att).hidden_states
        print("Finished forward pass!")

        for L in LAYERS:
            layer_out = outs[L]

            mask = torch.zeros_like(att, dtype=torch.bool)
            for i, n_sub in enumerate(lengths):
                mask[i, 1 : 1 + n_sub] = True

            masked = layer_out * mask.unsqueeze(-1)
            sum_vecs = masked.sum(dim=1)
            lengths_tensor = torch.tensor(lengths, device=device).unsqueeze(1)
            avg_vecs = sum_vecs / lengths_tensor

            store[f"layer{L}"][start : start + len(bw)] = avg_vecs.cpu().numpy()

print(f"Finished. Embeddings stored in {OUTPUT_H5}")
store.close()
