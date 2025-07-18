from pathlib import Path
import re
import torch
import itertools
import string
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

import h5py
import numpy as np

def keep_letters(text):
    return ''.join(ch for ch in text if ch.isalpha() or ch == ' ')


def calculate_dat_score(model, words, minimum=7):
    uniques_set = set()
    uniques = []

    for word in words:
        valid = model.validate(word)
        if valid and valid not in uniques_set:
            uniques.append(valid)
            uniques_set.add(valid)

    if len(uniques) < minimum:
        return None  # Not enough valid words
    subset = uniques[:minimum]

    # Compute DAT score: average pairwise distance × 100
    distances = [model.distance(w1, w2) for w1, w2 in itertools.combinations(subset, 2)]
    if not distances:
        return None

    return (sum(distances) / len(distances)) * 100.0

def calculate_dsi_score(model, sentences):
    test1 = [f"{sentence1}:{sentence2}" for sentence1, sentence2 in itertools.combinations(sentences, 2)]
    print(test1)
    model.distance(test1[0].split(":")[0], test1[0].split(":")[1])
    distances = [model.distance(sentence1, sentence2) for sentence1, sentence2 in itertools.combinations(sentences, 2)]
    if not distances:
        return None

    return (sum(distances) / len(distances)) * 100.0

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def validate(self, word):
        """Validate or transform a word. Return None or a transformed word."""
        pass

    @abstractmethod
    def distance(self, word1, word2):
        """Compute the distance between two validated words."""
        pass

class BERT_Encoder(BaseEmbeddingModel):
    def __init__(self, model="bert-base-uncased", layer=6, max_token_len=50):
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.filter_list = set(self.tokenizer.all_special_tokens) | set(string.punctuation)
        self.layer = layer
        self.max_token_len = max_token_len

    def validate(self, word):
        if isinstance(word, str) and word != "":
            return word
        return None

    def get_unit_vector(self, word):
        toks = self.tokenizer(word, max_length=self.max_token_len, padding="max_length", truncation=True, return_tensors="pt")

        with torch.no_grad():
            out  = self.model(**toks)
            hids = out.hidden_states[self.layer]

        dec_tokens = [self.tokenizer.decode([tid]) for tid in toks["input_ids"][0]]

        keep_idx = np.where(np.in1d(dec_tokens, list(self.filter_list), invert=True))[0]

        if len(keep_idx) == 0:
            vecs = hids[0, :1, :]
        else:
            vecs = hids[0, keep_idx, :]

        # mean‑pool & l2‑normalise
        pooled = vecs.mean(dim=0)
        return pooled / pooled.norm()

    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        return 1.0 - self.get_unit_vector(word1) @ self.get_unit_vector(word2)
    
    def __str__(self) -> str:
        return "BERT SuperClass"

class BERT_Encoder_L6(BERT_Encoder):
    def __init__(self, model="bert-base-uncased", max_token_len=50):
        super().__init__(model=model, layer=6, max_token_len=max_token_len)

    def __str__(self) -> str:
        return "BERT_ENCODER_L6"

class BERT_Encoder_L7(BERT_Encoder):
    def __init__(self, model="bert-base-uncased", max_token_len=50):
        super().__init__(model=model, layer=7, max_token_len=max_token_len)
    
    def __str__(self) -> str:
        return "BERT_ENCODER_L7"
    
class BERT_WordEmbeddings(BaseEmbeddingModel):
    def __init__(self, model="./models/bert_midlayer_dict.h5", dictionary="./models/words.txt", layer=6):
        with h5py.File(model, "r",
               rdcc_nbytes=512*1024**2,
               rdcc_nslots=200003,
               rdcc_w0=0.75) as f:

            tokens = f["tokens"].asstr()[:]
            token_to_index = {tok: i for i, tok in enumerate(tokens)}

            layer6 = torch.from_numpy(f["layer6"][:])
            layer6 = normalize(layer6, p=2, dim=1)
        
        word_pat = re.compile(r"^[a-z][a-z-]*[a-z]$")
        wanted, idxs = [], []

        with Path(dictionary).open(encoding="utf8") as fh:
            for w in fh:
                w = w.rstrip("\n")
                if word_pat.fullmatch(w) and w in token_to_index:
                    wanted.append(w)
                    idxs.append(token_to_index[w])

        vecs = layer6[idxs]
        self.vectors = {w: v for w, v in zip(wanted, vecs)}


    def validate(self, word):
        """Clean up word and find best candidate to use"""

        # Strip unwanted characters
        clean = clean_word(word)

        # Generate candidates for possible compound words
        candidates = []
        if clean:
            if " " in clean:
                candidates.append(re.sub(r" +", "-", clean))
                candidates.append(re.sub(r" +", "", clean))
            else:
                candidates.append(clean)
                if "-" in clean:
                    candidates.append(re.sub(r"-+", "", clean))
            for cand in candidates:
                if cand in self.vectors:
                    return cand # Return first word that is in model
        return None # Could not find valid word

    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        return 1.0 - self.vectors.get(word1) @ self.vectors.get(word2)
    
    def __str__(self) -> str:
        return "BERT_WordEmbeddings"

class BERT_WordEmbeddings_L6(BERT_WordEmbeddings):
    def __init__(self, model="./models/bert_midlayer_dict.h5", dictionary="./models/words.txt"):
        super().__init__(model=model, dictionary=dictionary, layer=6)

    def __str__(self) -> str:
        return "BERT_WordEmbeddings_L6"

class BERT_WordEmbeddings_L7(BERT_WordEmbeddings):
    def __init__(self, model="./models/bert_midlayer_dict.h5", dictionary="./models/words.txt"):
        super().__init__(model=model, dictionary=dictionary, layer=7)
    
    def __str__(self) -> str:
        return "BERT_WordEmbeddings_L7"


"""
Based on code from (https://github.com/other-user/other-repo), originally by jayolson. 
It was slightly modified to suit best the current codebase.
Modified under https://github.com/jayolson/divergent-association-task/blob/main/LICENSE.txt.
(Copyright 2021 Jay Olson; see LICENSE)
"""
def clean_word(word):
    # Strip unwanted characters
    clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
    if len(clean) <= 1:
        return None # Word too short
    return clean

class GloVe(BaseEmbeddingModel):
    def __init__(self, model="./models/glove.840B.300d.txt", dictionary="./models/words.txt", pattern="^[a-z][a-z-]*[a-z]$"):
        """Join model and words matching pattern in dictionary"""

        # Keep unique words matching pattern from file
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))

        # Join words with model
        vectors = {}
        with open(model, "r", encoding="utf8") as f:
            for line in tqdm(f, desc="Initializing model"):
                tokens = line.split(" ")
                word = tokens[0]
                if word in words:
                    vector = np.asarray(tokens[1:], "float32")
                    vectors[word] = vector / np.linalg.norm(vector)
        self.vectors = vectors

    def validate(self, word):
        """Clean up word and find best candidate to use"""

        # Strip unwanted characters
        clean = clean_word(word)

        # Generate candidates for possible compound words
        candidates = []
        if clean:
            if " " in clean:
                candidates.append(re.sub(r" +", "-", clean))
                candidates.append(re.sub(r" +", "", clean))
            else:
                candidates.append(clean)
                if "-" in clean:
                    candidates.append(re.sub(r"-+", "", clean))
            for cand in candidates:
                if cand in self.vectors:
                    return cand # Return first word that is in model
        return None # Could not find valid word

    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        return 1.0 - self.vectors.get(word1) @ self.vectors.get(word2)
    
    def __str__(self) -> str:
        return "GloVe"
