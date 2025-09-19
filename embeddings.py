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
    def __init__(self):
        print(f"Initializing {str(self)}")
    
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
    
    def __str__(self):
        return "Base Embedding Model"

class BERT_Encoder(BaseEmbeddingModel):
    """
    Base class for the BERT embedding model.
    """
    def __init__(self, model="bert-base-uncased", layer=6, max_token_len=50):
        super().__init__()
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
        super().__init__()
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
        super().__init__()

        # Keep unique words matching pattern from file
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))

        # Join words with model
        vectors = {}
        with open(model, "r", encoding="utf8") as f:
            for line in tqdm(f):
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
        super().__init__()
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
            
        vec = np.expand_dims(vec, axis=0)
        return mean_pool_l2_normalize_numpy(vec)

    def __str__(self) -> str:
        return "FastTextEmbeddings"

class Word2VecEmbeddings(BaseEmbeddingModel):
    """
    Embedding model wrapper for Word2VecEmbeddings.
    Model obtained from https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/refs/heads/master/GoogleNews-vectors-negative300.bin.gz
    """
    def __init__(self, model="./models/GoogleNews-vectors-negative300.bin", binary=True):
        super().__init__()
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
            if v is not None:
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
        
        if emb is not None:
            emb = np.expand_dims(emb, axis=0)
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
        super().__init__()
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


### In development
class GraniteMultilingualEmbeddingsAdvanced(BaseEmbeddingModel):
    def __init__(
            self,
            model_id: str = "ibm-granite/granite-embedding-278m-multilingual",
            device: str | None = None,
            max_token_len: int = 512,
            batch_size: int = 32,
            local_files_only: bool = False):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_token_len = int(max_token_len)
        self.batch_size = int(batch_size)

        # Load HF model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

        # Ensure we can pad if the tokenizer doesn't ship with a pad token
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "sep_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token

        self.dim = int(getattr(self.model.config, "hidden_size", 768))
        self._cache: dict[str, torch.Tensor] = {}

    def clean(self, word):
        return clean_word(word)

    def get_unit_vector(self, text: str) -> torch.Tensor:
        """
        Return an L2-normalized vector (torch.Tensor, shape [dim]).
        """
        if not isinstance(text, str) or not text.strip():
            return torch.zeros(self.dim, dtype=torch.float32)

        key = text 
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        vec = self._encode_batch([text])[0]
        self._cache[key] = vec
        return vec

    def __str__(self) -> str:
        return "GraniteMultilingualEmbeddings"

    @torch.inference_mode()
    def encode(self, texts: list[str], as_numpy: bool = False) -> torch.Tensor | np.ndarray:
        """
        Batch-embed a list of strings. Returns a 2D tensor of unit vectors.
        Also populates the in-memory cache.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32) if as_numpy else torch.empty((0, self.dim))

        # Identify which need computing
        to_compute = []
        order = []
        for t in texts:
            order.append(t)
            if t not in self._cache:
                to_compute.append(t)

        # Compute missing in batches
        for i in range(0, len(to_compute), self.batch_size):
            batch = to_compute[i : i + self.batch_size]
            reps = self._encode_batch(batch)
            for k, v in zip(batch, reps):
                self._cache[k] = v

        # Assemble outputs in requested order
        stacked = torch.stack([self._cache[t] for t in order], dim=0)
        return stacked.numpy() if as_numpy else stacked

    def precompute(self, words_or_sentences: list[str]) -> dict[str, np.ndarray]:
        """
        Batch-embed and cache. Returns a dict of {original_text: vector(np.float32)}.
        Useful to speed up DAT/DSI runs.
        """
        vecs = self.encode(words_or_sentences, as_numpy=True)
        return {t: v for t, v in zip(words_or_sentences, vecs)}

    # -------- Internal: model forward + pooling --------

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize -> forward -> CLS pooling -> L2-normalize.
        Returns CPU float32 tensor of shape [len(texts), dim].
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)

        # CLS pooling
        cls = out.last_hidden_state[:, 0, :]  # [B, dim]

        # L2-normalize
        cls = normalize(cls, p=2, dim=1)

        return cls.detach().cpu().to(torch.float32)


### In development
class GoogleAPIEmbedding(BaseEmbeddingModel):
    """
    Google Gemini Embedding wrapper (gemini-embedding-001) with HDF5 caching.

    - Use `precompute(words)` to batch-embed and cache a list of words first.
    - On `distance()` or `get_unit_vector()`, if an item is missing in cache,
      it will be embedded on the fly and appended to the cache file.
    - Embeddings are L2-normalized before storage to keep distances consistent.

    Requirements:
        pip install google-genai h5py

    Environment:
        GOOGLE_API_KEY must be set for the Google GenAI client to authenticate.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        batch: bool = True,
        cache_path: str = "./models/googleapi_embeddings.h5",
        output_dimensionality: int = 100,
        batch_size: int = 100,
    ):
        try:
            from google import genai  # lazy import to give nice error if missing
            from google.genai import types as genai_types
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "google-genai package not installed. Install with: pip install google-genai"
            ) from e

        self._genai = genai
        self._genai_types = genai_types
        self.client = genai.Client()

        self.model = model
        self.batch = batch
        self.cache_path = Path(cache_path)
        self.output_dim = int(output_dimensionality)
        self.batch_size = int(batch_size)

        # in-memory index for fast checks; maps token -> row index
        self._index = {}
        # initialize store (create if needed) and load existing index
        self._init_store()
        self._load_index()

    def clean(self, word):
        """Clean up word: keep letters/spaces/hyphens; lower-case; min length 2."""
        return clean_word(word)

    def get_unit_vector(self, word) -> np.ndarray:
        """
        Return an L2-normalized embedding vector (np.ndarray, float32).
        If not in cache, embed now, store, and return.
        """
        if not isinstance(word, str):
            return np.zeros(self.output_dim, dtype=np.float32)

        clean = self.validate(word)
        if not clean:
            return np.zeros(self.output_dim, dtype=np.float32)

        vec = self._lookup(clean)
        if vec is not None:
            return vec

        # Not cached: embed and store
        new = self._embed_words([clean])
        self._append_to_store(list(new.keys()), np.vstack(list(new.values())))
        return new[clean]

    def distance(self, word1, word2) -> float:
        """Cosine distance (1 - cosine similarity) between two words."""
        v1 = self.get_unit_vector(word1)
        v2 = self.get_unit_vector(word2)
        # Both are unit vectors; dot is cosine similarity
        return float(1.0 - float(np.dot(v1, v2)))

    def __str__(self) -> str:
        return "GoogleAPIEmbedding"

    # ---------- Public convenience API ----------

    def precompute(self, words):
        """
        Batch-embed and cache a list of words/phrases.
        Returns {word: vector} *for the validated inputs only*.
        """
        if not words:
            return {}

        # Clean, filter, dedupe
        cleaned = [self.validate(w) for w in words if isinstance(w, str)]
        cleaned = [w for w in cleaned if w]  # drop None
        # de-dup but preserve order
        seen = set()
        todo = []
        for w in cleaned:
            if w not in seen:
                seen.add(w)
                todo.append(w)

        # Exclude already-cached
        missing = [w for w in todo if w not in self._index]

        if not missing:
            # Just return what we have (read from cache)
            return {w: self._lookup(w) for w in todo}

        # Embed missing in batches
        new_map = self._embed_words(missing)
        if new_map:
            self._append_to_store(list(new_map.keys()), np.vstack(list(new_map.values())))

        # Return full map for requested (cached + newly added)
        result = {w: self._lookup(w) for w in todo if self._lookup(w) is not None}
        return result

    # ---------- HDF5 cache helpers ----------

    def _init_store(self):
        """
        Ensure the HDF5 file exists and has resizable datasets:
          - 'tokens': 1D variable-length UTF-8 strings
          - 'vectors': 2D float32 with shape (N, output_dim)
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            with h5py.File(self.cache_path, "w") as f:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(
                    "tokens",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=str_dtype,
                    chunks=True,
                )
                f.create_dataset(
                    "vectors",
                    shape=(0, self.output_dim),
                    maxshape=(None, self.output_dim),
                    dtype="float32",
                    chunks=True,
                )

    def _load_index(self):
        """Load token -> row index into memory for quick lookups."""
        with h5py.File(self.cache_path, "r") as f:
            tokens = list(f["tokens"].asstr()[:])
        self._index = {tok: i for i, tok in enumerate(tokens)}

    def _lookup(self, token: str) -> np.ndarray | None:
        """Return a cached vector if present; else None."""
        idx = self._index.get(token)
        if idx is None:
            return None
        with h5py.File(self.cache_path, "r") as f:
            vec = f["vectors"][idx, :]
        # Ensure float32 numpy
        v = np.asarray(vec, dtype=np.float32)
        # Stored vectors are already normalized; return as-is
        return v

    def _append_to_store(self, tokens: list[str], vectors: np.ndarray):
        """
        Append new (token, vector) rows to the HDF5 datasets and update index.
        `vectors` is expected to be (len(tokens), output_dim) float32 and normalized.
        """
        if not tokens:
            return
        assert vectors.shape[0] == len(tokens), "Length mismatch for tokens/vectors"
        assert vectors.shape[1] == self.output_dim, "Vector dimensionality mismatch"

        with h5py.File(self.cache_path, "a") as f:
            tok_ds = f["tokens"]
            vec_ds = f["vectors"]

            old_n = tok_ds.shape[0]
            new_n = old_n + len(tokens)

            tok_ds.resize((new_n,))
            vec_ds.resize((new_n, self.output_dim))

            tok_ds[old_n:new_n] = np.array(tokens, dtype=object)
            vec_ds[old_n:new_n, :] = vectors.astype("float32", copy=False)

        # update in-memory index
        for i, t in enumerate(tokens):
            self._index[t] = len(self._index)  # sequential after append

    # ---------- Embedding (Google GenAI) ----------

    def _embed_words(self, words: list[str]) -> dict[str, np.ndarray]:
        """
        Call the Google GenAI embed API for a list of already-cleaned words.
        Returns a dict {word: normalized_vector (float32)}.
        """
        if not words:
            return {}

        cfg = self._genai_types.EmbedContentConfig(output_dimensionality=self.output_dim)

        out: dict[str, np.ndarray] = {}
        if self.batch:
            for chunk in tqdm(self._batch_list(words, self.batch_size), total=len(words)//self.batch_size):
                res = self.client.models.embed_content(
                    model=self.model,
                    contents=chunk,
                    config=cfg,
                )
                # The SDK returns an object with a `.embeddings` list
                vecs = [np.asarray(e.values, dtype=np.float32) if hasattr(e, "values") else np.asarray(e, dtype=np.float32)
                        for e in res.embeddings]
                vecs = [self._unit(v) for v in vecs]
                for w, v in zip(chunk, vecs):
                    out[w] = v
        else:
            # one-by-one (slower)
            for w in words:
                res = self.client.models.embed_content(
                    model=self.model,
                    contents=[w],
                    config=cfg,
                )
                v = res.embeddings[0].values if hasattr(res.embeddings[0], "values") else res.embeddings[0]
                v = np.asarray(v, dtype=np.float32)
                out[w] = self._unit(v)

        return out

    @staticmethod
    def _batch_list(lst, batch_size):
        """Yield successive batch_size-sized chunks from lst."""
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector (float32)."""
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n
