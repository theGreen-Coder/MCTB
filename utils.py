import re
import itertools
import numpy as np
import torch
import unicodedata

def keep_letters(text):
    """Takes a string and returns only alphabetic characters."""
    return ''.join(ch for ch in text if ch.isalpha() or ch == ' ')

def keep_numbers(text: str) -> str:
    return ''.join(ch for ch in text if ch.isdigit())

def clean_word(word):
    """
    Cleans a word by removing non-alphabetic characters (except hyphens and spaces) and converts to lowercase.
    Returns None if the cleaned word is too short (<= 1 character).
    """
    if not word:
        return None
    
    clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
    if len(clean) <= 1:
        return None # Word too short
    return clean

def clean_word_unicode(word):
    """
    Keep letters from any script (Unicode), plus hyphens and spaces.
    Remove digits, underscores, and other punctuation/symbols.
    Return None if cleaned is empty. If the cleaned result is a single character,
    allow it only when it's a 'Lo' (Letter, other) character (e.g., CJK/Kana).
    """
    if not word:
        return None

    # Normalize to handle full-width/compatibility forms consistently
    word = unicodedata.normalize("NFKC", word)

    # Keep Unicode word chars, spaces, and hyphens; then remove digits/underscores
    clean = re.sub(r"[^\w\s-]|[\d_]", "", word).strip().lower()

    if not clean:
        return None

    if len(clean) == 1:
        # Allow single-character words only if they are 'Lo' (e.g., 中, 時, あ, カ)
        return clean if unicodedata.category(clean) == "Lo" else None

    return clean


def calculate_dat_score(model, words, minimum=7, maximum=12):
    """
    Cleans words according to model preferences. If less than 7 words can be clean, returns None.
    Otherwise, returns the average pairwise distance between all the word embeddings (with a maximum of 12 words).
    """
    if words is None:
        print("WARNING: Could not calculate DAT: no words were provided!")
        return None
    uniques_set = set()
    uniques = []

    for word in words:
        if word != "":
            valid = model.clean(word)
            if valid and valid not in uniques_set:
                uniques.append(valid)
                uniques_set.add(valid)

    if len(uniques) < minimum:
        print("WARNING: Could not calculate DAT: not enough valid words!")
        return None  # Not enough valid words
    subset = uniques[:maximum]

    # Compute DAT score: average pairwise distance × 100
    distances = [
        d
        for w1, w2 in itertools.combinations(subset, 2)
        if (d := model.distance(w1, w2)) is not None
    ]

    if not distances:
        print("WARNING: Could not calculate DAT: distances were not properly calculated!")
        return None

    return (sum(distances) / len(distances)) * 100.0

def mean_pool_l2_normalize_torch(vectors, eps=1e-12):
    """Mean pooling and L2 normaization on a torch tensor"""
    pooled = vectors.mean(dim=0)
    norm = pooled.norm(p=2)
    return pooled / torch.clamp(norm, min=eps)

def mean_pool_l2_normalize_numpy(vectors, eps=1e-12):
    """Mean pooling and L2 normaization on a numpy array"""
    pooled = np.mean(vectors, axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / max(norm, eps)

def calculate_dsi_score(model, sentences, minimum=4, maximum=10):
    """
    Calcualte the pairwise distance between sentences in DSI.
    If no distances are given or not enough sentences are given, returns None.
    Otherwise, it returns the average pairwise distance between all the sentences of the story.
    """
    if not distances:
        return None
    
    if len(sentences) < minimum:
        return None  # Not enough sentences
    sentences = sentences[:maximum]
    
    distances = [model.distance(sentence1, sentence2) for sentence1, sentence2 in itertools.combinations(sentences, 2)]
    
    return (sum(distances) / len(distances)) * 100.0

def generate_configs(start, end, step):
    """
    Generates temperature variable configs from start to end.
    Used in testing to see the effect of temperature in the answers.
    """
    configs = []
    temperature = start
    while temperature <= end + 1e-9:  # small tolerance for floating point issues
        configs.append({"temperature": round(temperature, 2)})
        temperature += step
    return configs