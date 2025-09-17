import json
import random
import re
from typing import List
from tqdm import tqdm
from embeddings import BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from collections import defaultdict
from request import Request, run_request
from creative_tests import DivergentAssociationTest
from utils import *

class HardDivergentAssociationTest(DivergentAssociationTest):
    def __init__(self, models, configs, embedding_models=[GloVe, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7], common=True, repeats=1, delay=0, n_words=25, given_words=50):
        super().__init__(models=models, configs=configs, embedding_models=embedding_models, repeats=repeats, delay=delay, n_words=n_words)
        self.given_words = given_words
        self.common = common
    
    def get_n_random_words(self, dictionary="./models/words.txt"):
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match("^[a-z][a-z-]*[a-z]$", line):
                    words.add(line.rstrip("\n"))

        unique_words = list(words)
        random_words = []

        for _ in range(self.given_words):
            random_words.append(random.choice(unique_words))

        return random_words
    
    def set_prompt(self, letter: str, random_words: List):
        self.test_prompt = (
            f"Please enter {str(self.n_words)} words, each starting with the letter '{letter}', that are as different from each other as possible, in all meanings and uses of the words. "
            f"Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). "
            f"No specialized vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).\n"
            f"Your goal is to create words so that, no combination of any two words should share anything in common, have shared threads, connections, or fit into a specific category."
            f"Make a list of these {str(self.n_words)} words, a single word in each entry of the list. Output just these numbered words, no comments, nothing before or after the list.\n."
            f"In addition to starting with '{letter}' and being different from each other, the words should also be as different as possible from the following {self.given_words} words provided:"
        )
        self.test_prompt += "\n".join(random_words) 
    
    def __str__(self):
        return "Hard" + super().__str__()
    
    def request(self) -> dict:
        non_clean_llm_response = []

        for letter in tqdm("abcdefghiklmnoprstuwy"):
        # for letter in tqdm("ab"):
            for repeat in range(self.repeats):
                if self.common:
                    random_words = self.get_n_random_words(dictionary="models/5k_common.txt")
                else:
                    random_words = self.get_n_random_words()
                self.set_prompt(letter=letter, random_words=random_words)

                HardDAT_request = Request(
                    models=self.models,
                    prompt=self.test_prompt,
                    configs=self.configs,
                    repeats=1,
                    delay=self.delay,
                    verbose=False
                )

                llm_response = run_request(HardDAT_request)
                llm_response["config"] = {}
                llm_response["config"]["given_words"] = random_words
                llm_response["config"]["letter"] = letter
                llm_response["config"]["repeat"] = repeat

                non_clean_llm_response.append(llm_response)
        
        with open(f"responses/{str(self)}.json", "w") as json_file:
            json.dump(non_clean_llm_response, json_file, indent=4)
        
        return non_clean_llm_response

    def clean_llm_response(self, prev: dict | str) -> dict:
        # Check input
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                non_clean_llm_response = json.load(file)
            self.set_id(prev)

        elif isinstance(prev, dict) or isinstance(prev, List):
            non_clean_llm_response = prev
        

        merged = defaultdict(lambda: defaultdict(list))

        for entry in non_clean_llm_response:
            random_words = entry["config"]["given_words"]
            letter = entry["config"]["letter"]
            repeat = entry["config"]["repeat"]

            for model_key in self.models:
                if model_key in entry:
                    for temp_key, word_lists in entry[model_key].items():
                        # all_words = [clean_response(word) for word in word_lists]
                        # all_words += random_words
                        assert len(word_lists) == 1 and len(word_lists[0]) == 1
                        merged[model_key][temp_key].extend([self.clean_response(word_lists[0][0]) + random_words])

        # Optional: Convert defaultdicts to dicts
        final_result = {model: dict(temps) for model, temps in merged.items()}

        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            json.dump(final_result, json_file, indent=4)

        return final_result
    
    def run(self):
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        return self.calculate_embeddings(prev=prev)
        
        
        
