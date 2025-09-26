import re
import json
from typing import List
import numpy as np
from embeddings import BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from request import Request, run_request
from scipy.stats import norm
from datetime import datetime
from utils import *

class DivergentAssociationTest():
    def __init__(self, models, configs, embedding_models=[GloVe, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7], repeats=0, delay=0, n_words=10, standard_prompt=True, starts_with=None, file_name=None):
        self.models = models
        self.configs = configs
        self.repeats = repeats
        self.embedding_models = embedding_models
        self.id = str(datetime.now().strftime("%m%d%H%M%S"))
        self.delay = delay
        self.n_words = n_words
        self.file_name = file_name

        self.addition_specs = ''
        if starts_with:
            self.addition_specs += f"All words must start with the letter {starts_with}."

        # Test is available sampling from several words or just a standard prompt
        if standard_prompt:
            self.test_prompt = (
                f"Please enter {str(self.n_words)} words that are as different from each other as possible, in all meanings and uses of the words. "
                f"Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). "
                f"No specialized vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). "
                f"Make a list of these {str(self.n_words)} words, a single word in each entry of the list. Do not write anything else but the {str(self.n_words)} words." + " " + self.addition_specs
            )
        else:
            self.test_prompt = [
                (f"Please enter {str(self.n_words)} words that are as different from each other as possible, in all meanings and uses of the words. "
                f"Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). "
                f"No specialized vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). "
                f"Make a list of these {str(self.n_words)} words, a single word in each entry of the list. Do not write anything else but the {str(self.n_words)} words." + " " + self.addition_specs), # 1st prompt

                (f"Please enter {str(self.n_words)} English nouns that are as different as possible in meaning and use — no proper nouns, "
                f"no technical terms, only single common nouns you think of yourself. List exactly {str(self.n_words)}, one word per entry, and nothing else." + " " + self.addition_specs), # 2nd prompt

                (f"List {str(self.n_words)} common, single-word English nouns that are maximally different from one another. "
                f"Do not use proper nouns, technical terms, or words for objects in your immediate surroundings. Output only the {str(self.n_words)} words in a list." + " " + self.addition_specs), # 3rd prompt

                (f"Enter {str(self.n_words)} English nouns that are as different from each other as possible in meaning and usage. "
                f"Rules: Each word must be a single noun (no proper nouns, technical terms, or multi-word phrases). Choose words independently, without relying on nearby objects. "
                f"List them with one word per entry, without any additional text." + " " + self.addition_specs), # 4th prompt

                (f"List {str(self.n_words)} distinct common English nouns (things, objects, concepts). Rules: Single words only. "
                f"No proper nouns, technical terms, or words from your surroundings. Output only the list." + " " + self.addition_specs) # 5th prompt
            ]

    
    def __str__(self):
        if self.file_name is not None and isinstance(self.file_name, str):
            return self.file_name
        return "DAT_"+str(self.id)+"_"+str(len(self.models))+"models_"+str(len(self.configs))+"configs_"+str(self.n_words)+"words"
    
    def set_id(self, filename):
        match = re.search(r'_(\d{10})_', filename)
        if match:
            file_id = match.group(1)
            print(f"Found id {file_id}")
            self.id = file_id
        else:
            print("ID not found")
    
    def init_word_embeddings(self):
        initialized = []

        for embedding_model in self.embedding_models:
            initialized.append(embedding_model())
        
        self.embedding_models = initialized
    
    def clean_response(self, response):
        def keep_letters(text: str) -> str:
            return ''.join(ch for ch in text if ch.isalpha() or ch == ' ')
        
        return_words = []
        # Check for thought included responses (very specific case)
        if response is not None and isinstance(response, list) and len(response) == 2 and response[0].lower().startswith("thought\n"):
            response = response[1]
        new_line_words = response.split("\n")

        if len(new_line_words) >= 7:
            for word in new_line_words:
                if word != "" and len(word) <= 15:
                    clean_word = keep_letters(word)

                    if len(clean_word.split()) <= 1:
                        return_words.append(clean_word.strip())
            return return_words
        
        return_words = []
        comma_separated_words = response.split(",")

        if len(comma_separated_words) >= 7:
            for word in comma_separated_words:
                if word != "" and len(word) <= 15:
                    clean_word = keep_letters(word)

                    if len(clean_word.split()) <= 1:
                        return_words.append(clean_word.strip())
            return return_words

        return None
    
    def request(self) -> dict:
        DAT_request = Request(
            models=self.models,
            prompt=self.test_prompt,
            configs=self.configs,
            repeats=self.repeats,
            default_delay=self.delay
        )

        llm_response = run_request(DAT_request)

        with open(f"responses/{str(self)}.json", "w") as json_file:
            json.dump(llm_response, json_file, indent=4)
        
        return llm_response

    def clean_llm_response(self, prev: dict | str) -> dict:
        # Check input
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                llm_response = json.load(file)
            self.set_id(prev)

        elif isinstance(prev, dict):
            llm_response = prev
        
        llm_response_clean = llm_response
        for model, configs in llm_response.items():
            for config, repeats in configs.items():
                for idx, repeat in enumerate(repeats):
                    response = repeat[0]

                    clean_response = self.clean_response(response=response)
                    llm_response_clean[model][config][idx] = clean_response if clean_response else ""
        
        # Export cleaned responses results
        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            json.dump(llm_response_clean, json_file, indent=4)
        
        return llm_response_clean

    def calculate_embeddings(self, prev) -> List:
        return_files = []
        # Check input
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                llm_response_clean = json.load(file)
            self.set_id(prev)

        elif isinstance(prev, dict):
            llm_response_clean = prev
        
        results = {}
        model_distribution = {}

        self.init_word_embeddings()

        for model, configs in llm_response_clean.items():
            for config, repeats in configs.items():
                model_key = f"{model}_{str(config)}"

                for embedding_model in self.embedding_models:
                    emb_key = str(embedding_model)
                    scores = []

                    for repeat in repeats:
                        # 4-5. Get embeddings from model and calculate cosine similarity
                        if len(repeat) > 0:
                            score = calculate_dat_score(embedding_model, repeat)
                            if score:
                                scores.append(float(score))
                        else:
                            print(f"No response was found for one of the responses.")

                    # store per‑model/config
                    results.setdefault(model_key, {})[emb_key] = scores
                    # accumulate global list for this embedding
                    model_distribution.setdefault(emb_key, []).extend(scores)
                results.setdefault(model_key, {})["config"] = json.loads(config)
        
        with open(f"results/{str(self)}_unnormalized.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
            return_files.append(f"results/{str(self)}_unnormalized.json")

        # Compute normalized results (in case of several embedding models)
        stats = {}
        for emb_key, all_scores in model_distribution.items():
            a = np.asarray(all_scores)
            mean, std = a.mean(), a.std()
            stats[emb_key] = (mean, std if std > 0 else None)

        normalized_results = {}
        for model_key, emb_dict in results.items():
            normalized_results[model_key] = {}

            for emb_key, raw_scores in emb_dict.items():
                if emb_key != "config":
                    mean, std = stats[emb_key]

                    if std is None:
                        normed = [50.0] * len(raw_scores)
                    else:
                        normed = [
                            norm.cdf((s - mean) / std) * 100.0  # Z‑score → CDF → 0‑100
                            for s in raw_scores
                        ]

                    normalized_results[model_key][emb_key] = normed
                else:
                    normalized_results[model_key]["config"] = raw_scores

        with open(f"results/{str(self)}_normalized.json", "w") as json_file:
            json.dump(normalized_results, json_file, indent=4)
            return_files.append(f"results/{str(self)}_normalized.json")
        
        return return_files
    
    """
        Process:
        1. DAT Request -> 2. LLM Response -> 3. Text Splitter -> 4. Embeddings -> 5. Calculate Cosine Similarity -> 6. Export results
    """

    def run(self):
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        return self.calculate_embeddings(prev=prev)
    