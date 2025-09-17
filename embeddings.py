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

from utils import *

class BaseEmbeddingModel(ABC):
    """
    Base class for all embedding models in this repository.
    """
    @abstractmethod
    def clean(self, word):
        """Cleans string based on model preferences. Returns cleaned word or None if word cannot be processed/cleaned."""
        pass

    @abstractmethod
    def distance(self, word1, word2):
        """Compute the distance between two cleaned words. """
        pass
    
    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two sentence/words"""
        return 1.0 - self.get_unit_vector(word1) @ self.get_unit_vector(word2)

class BERT_Encoder(BaseEmbeddingModel):
    """
    Base class for the BERT embedding model.
    """
    def __init__(self, model="bert-base-uncased", layer=6, max_token_len=50):
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.filter_list = set(self.tokenizer.all_special_tokens) | set(string.punctuation)
        self.layer = layer
        self.max_token_len = max_token_len

    def clean(self, text):
        if isinstance(text, str) and text != "":
            return text
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

        return mean_pool_l2_normalize_torch(vecs)
    
    def __str__(self) -> str:
        return "BERT_BASE_CLASS"

class BERT_Encoder_L6(BERT_Encoder):
    """
    BERT Encoder using embeddings from layer 6.
    """
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
            self.layer = layer

            chosen_layer = torch.from_numpy(f[f"layer{self.layer}"][:])
            chosen_layer = normalize(chosen_layer, p=2, dim=1)
        
        word_pat = re.compile(r"^[a-z][a-z-]*[a-z]$")
        wanted, idxs = [], []

        with Path(dictionary).open(encoding="utf8") as fh:
            for w in fh:
                w = w.rstrip("\n")
                if word_pat.fullmatch(w) and w in token_to_index:
                    wanted.append(w)
                    idxs.append(token_to_index[w])

        vecs = chosen_layer[idxs]
        self.vectors = {w: v for w, v in zip(wanted, vecs)}

    def clean(self, word):
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

class GloVe(BaseEmbeddingModel):
    """
    Based on code from (https://github.com/other-user/other-repo), originally by jayolson. 
    It was slightly modified to suit best the current codebase.
    Modified under https://github.com/jayolson/divergent-association-task/blob/main/LICENSE.txt.
    (Copyright 2021 Jay Olson; see LICENSE)
    """
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

    def clean(self, word):
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

class FastTextEmbeddings(BaseEmbeddingModel):
    """
    Open-source, free, lightweight embedding model developed by Facebook.
    Model obtaind from https://huggingface.co/elishowk/fasttext_test2
    """
    def __init__(self, model="./models/cc.en.300.bin", lowercase=True):
        try:
            import fasttext
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "fasttext package not installed. Try: pip install fasttext-wheel (or pip install fasttext)"
            ) from e

        self._ft = fasttext.load_model(model)
        self.lowercase = lowercase

    def clean(self, word):
        clean = clean_word(word)
        if len(clean) <= 1:
            return None
        return clean

    def get_unit_vector(self, text: str) -> torch.Tensor:
        """
        Returns an L2-normalized torch vector for a word or sentence.
        """
        # Decide whether to treat as a "sentence" for averaging
        if " " in text:
            vec = self._ft.get_sentence_vector(text)
        else:
            vec = self._ft.get_word_vector(text)

        return mean_pool_l2_normalize_numpy(vec)

    def __str__(self) -> str:
        return "FastTextEmbeddings"

class Word2VecEmbeddings(BaseEmbeddingModel):
    """
    Embedding model wrapper for Word2VecEmbeddings.
    Model obtained from https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/refs/heads/master/GoogleNews-vectors-negative300.bin.gz
    """
    def __init__(self, model="./models/GoogleNews-vectors-negative300.bin", binary=True):
        try:
            from gensim.models import KeyedVectors
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "gensim not installed. Install with: pip install gensim"
            ) from e

        self.model = KeyedVectors.load_word2vec_format(model, binary=binary)
        try:
            self.model.fill_norms() # Precompute norms (gensim >= 4) so kv.get_vector(..., norm=True) is fast
        except Exception:
            pass  # older gensim versions may not need this / may not have fill_norms()

    def get_word_variants(self, text: str):
        """Generate plausible key variants (underscores, case variants, hyphen handling)."""
        if not isinstance(text, str):
            return []
        raw = text.strip()
        base = clean_word(raw)
        candidates = []

        # Generate candidates for possible compound words
        if base:
            if " " in base:
                candidates += [
                    base.replace(" ", "_"),
                    base.replace(" ", ""),
                    base.replace(" ", "-"),
                ]
            else:
                candidates.append(base)
                if "-" in base:
                    candidates += [
                        base.replace("-", "_"),
                        base.replace("-", ""),
                    ]

        # Try even more variants with different lowercase/uppercase capitalizations
        more = []
        for c in list(candidates):
            if "_" in c:
                parts = c.split("_")
                more += [
                    "_".join(p.capitalize() for p in parts),
                    "_".join(p.title() for p in parts),
                    "_".join(p.upper() for p in parts),
                ]
            else:
                more += [c.capitalize(), c.title(), c.upper()]

        if raw:
            more.append(raw) # Try the raw (user-provided) string

        # De-duplicate preserving order
        seen, out = set(), []
        for c in candidates + more:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def clean(self, text: str):
        if text and text != "":
            return clean_word(text)
        return None
    
    def try_different_variants(self, variants):
        if variants:
            for var in variants:
                try:
                    v = self.model.get_vector(var)
                    break
                except:
                    continue
            return v
        return None
        
    def get_unit_vector(self, text: str) -> np.ndarray:
        """Get an L2-normalized vector for a word or a mean-pooled sentence."""
        # Sentence embedding
        if isinstance(text, str) and " " in text:
            tokens = [t for t in re.split(r"\s+", clean_word(text) or "") if t]
            vecs = []
            
            for tok in tokens:
                variants = self.get_word_variants(tok)
                emb = self.try_different_variants(variants)
                if emb:
                    vecs.append(emb)
            if vecs:
                return mean_pool_l2_normalize_numpy(np.mean(np.vstack(vecs), axis=0))

        # Word/Token embedding
        variants = self.get_word_variants(text)
        emb = self.try_different_variants(variants)
        
        if emb:
            return mean_pool_l2_normalize_numpy(emb)
        return None

    def __str__(self) -> str:
        return "Word2VecEmbeddings"

class GraniteMultilingualEmbeddings(BaseEmbeddingModel):
    """
    Wrapper for IBM Granite multilingual embedding model.
    Returns CLS token embeddings, L2-normalized.
    """
    def __init__(
                self,
                model_id: str = "ibm-granite/granite-embedding-278m-multilingual",
                max_token_len: int = 512,
                device: str | None = None,
                local_files_only: bool = False,
            ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_token_len = max_token_len

        # Load model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.sep_token
            )

        self.dim = int(getattr(self.model.config, "hidden_size", 768))

    def clean(self, word):
        return clean_word(word)

    @torch.inference_mode()
    def get_unit_vector(self, text: str) -> torch.Tensor:
        """
        Return a single L2-normalized vector (torch.Tensor, shape [dim]).
        """
        if not isinstance(text, str) or not text.strip():
            return torch.zeros(self.dim, dtype=torch.float32)

        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)

        # CLS pooling + L2 normalize
        cls = out.last_hidden_state[:, 0, :]
        cls = normalize(cls, p=2, dim=1)

        return cls[0].detach().cpu().to(torch.float32)

    def __str__(self) -> str:
        return "GraniteMultilingualEmbeddings"


