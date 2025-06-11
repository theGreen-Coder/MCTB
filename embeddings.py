import re
import itertools
import numpy as np
from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def validate(self, word):
        """Validate or transform a word. Return None or a transformed word."""
        pass

    @abstractmethod
    def distance(self, word1, word2):
        """Compute the distance between two validated words."""
        pass

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
            for line in f:
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
