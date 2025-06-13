import os
import json
import numpy as np
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import GloVe, calculate_dat_score
from request import Request, run_request
from scipy.stats import norm
from datetime import datetime

class DivergentAssociationTest():
    def __init__(self, models, configs, embedding_models=GloVe(), repeats=0, delay=0):
        self.models = models
        self.configs = configs
        self.repeats = repeats
        self.embedding_models = [embedding_models] # Temporal for now (in the future will allow to do a list)
        self.creation_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.delay = delay

        self.test_prompt = 'Please enter 10 words that are as different from each other as possible, ' \
                            'in all meanings and uses of the words. Rules: Only single words in English. ' \
                            'Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). ' \
                            'No specialized vocabulary (e.g., no technical terms). ' \
                            'Think of the words on your own (e.g., do not just look at objects in your surroundings). ' \
                            'Make a list of these 10 words, a single word in each entry of the list.' \
                            'Do not write anything else, but the 10 words.'
    
    def __str__(self):
        model_list = '_'.join(self.models) if self.models else 'None'

        return "DAT_"+model_list+"_"+str(self.creation_time)
    
    def clean_response(self, response):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=0,
            chunk_overlap=0,
            separators=["\n", " "],
            keep_separator=False,
        )

        words = splitter.split_text(response)
        clean_words = [w for w in words if w.isalpha()]

        return clean_words

    """
        Process:
        1. DAT Request -> 2. LLM Response -> 3. Text Splitter -> 4. Embeddings -> 5. Calculate Cosine Similarity -> 6. Export results
    """
    def run(self, clean_response_file=None):
        if not clean_response_file:
            #################################################################
            # 1. Get the request ready
            #################################################################
            DAT_request = Request(
                models=self.models,
                prompt=self.test_prompt,
                configs=self.configs,
                repeats=self.repeats,
                delay=self.delay
            )

            #################################################################
            # 2. Get the LLM's response
            #################################################################
            llm_response = run_request(DAT_request)

            # Export responses results
            with open(f"responses/{str(self)}.json", "w") as json_file:
                json.dump(llm_response, json_file, indent=4)

            #################################################################
            # 3. Clean LLM's response
            #################################################################
            llm_response_clean = llm_response

            for model, configs in llm_response.items():
                for config, repeats in configs.items():
                    for idx, repeat in enumerate(repeats):
                        response = repeat[0]
                        llm_response_clean[model][config][idx] = self.clean_response(response=response)
            
            # Export cleaned responses results
            with open(f"responses/{str(self)}_clean.json", "w") as json_file:
                json.dump(llm_response_clean, json_file, indent=4)
        
        else:
            with open(clean_response_file, 'r') as file:
                llm_response_clean = json.load(file)

        #################################################################
        # 4-5. Get embeddings from model and calculate cosine similarity
        #################################################################
        results = {}
        model_distribution = {}

        for model, configs in llm_response_clean.items():
            for config, repeats in configs.items():
                model_key = f"{model}_{config}"

                for embedding_model in self.embedding_models:
                    emb_key = str(embedding_model)
                    scores = []

                    for repeat in repeats:
                        # 4-5. Get embeddings from model and calculate cosine similarity
                        if len(repeat) > 0:
                            score = calculate_dat_score(embedding_model, repeat)
                            scores.append(float(score))
                        else:
                            print(f"No response was found for one of the responses.")

                    # store per‑model/config
                    results.setdefault(model_key, {})[emb_key] = scores
                    # accumulate global list for this embedding
                    model_distribution.setdefault(emb_key, []).extend(scores)
        
        with open(f"results/{str(self)}_unnormalized.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

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
                mean, std = stats[emb_key]

                if std is None:
                    normed = [50.0] * len(raw_scores)
                else:
                    normed = [
                        norm.cdf((s - mean) / std) * 100.0  # Z‑score → CDF → 0‑100
                        for s in raw_scores
                    ]

                normalized_results[model_key][emb_key] = normed
        
        #################################################################
        # 6. Export results
        #################################################################
        with open(f"results/{str(self)}_normalized.json", "w") as json_file:
            json.dump(normalized_results, json_file, indent=4)

    