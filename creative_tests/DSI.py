import re
import json
import string
from typing import List, Optional
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, SBERT_Encoder
from request import Request, run_request
from scipy.stats import norm
from datetime import datetime
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import pandas as pd
import string
import time
import torch
from utils import *
import nltk
from nltk.tokenize import sent_tokenize
import itertools

class DivergentSemanticIntegration():
    def __init__(self, models, configs, repeats=1, delay=0, variant_prompts: Optional[list] = None):
        self.models = models
        self.configs = configs
        self.repeats = repeats
        self.id = str(datetime.now().strftime("%m%d%H%M%S"))
        self.delay = delay
        self.return_files = []
        self.filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])
        
        # DSI specific models
        self.segmenter = PunktSentenceTokenizer()
        
        # Add prompts to DSI
        self.prompts = []
        self.selected_words = []

        # Low/High semantic distance words
        low_semantic_words = ["stamp, letter, send", "belief, faith, sing", "petrol, diesel, pump", "year, week, embark"]
        high_semantic_words = ["stamp, letter, send", "gloom, payment, exist", "organ, empire, comply", "statement, stealth, detect"]
        
        for words in low_semantic_words:
            self.prompts.append(f"Please write a five-sentence creative story with the following three-word prompt: {words}. Please include all three words, be creative and imaginative when writing the sort story. Do not write anything else, but the story.")
            self.selected_words.append(keep_letters(words).split())
        
        for words in high_semantic_words:
            self.prompts.append(f"Please write a five-sentence creative story with the following three-word prompt: {words}. Please include all three words, be creative and imaginative when writing the sort story. Do not write anything else, but the story.")
            self.selected_words.append(keep_letters(words).split())
        
        if isinstance(variant_prompts, list):
            tmp_words = self.selected_words.copy()
            for v_prompt in variant_prompts:
                for tmp_w in tmp_words:
                    f_prompt = v_prompt.replace("||words||", ", ".join(tmp_w))
                    self.prompts.append(f_prompt)
                    self.selected_words.append(tmp_w)

        self.init_word_embeddings()
    
    def init_word_embeddings(self):      
        # Also init sentence splitter
        nltk.download("punkt")
        
        # Init bert-large-uncased
        print("Init large bert uncased")
        self.model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states = True) # initialize BERT model instance
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') # initialize BERT tokenizer
    
    def set_id(self, filename):
        match = re.search(r'_(\d{10})_', filename)
        if match:
            file_id = match.group(1)
            print(f"Found id {file_id}")
            self.id = file_id
        else:
            print("ID not found")
    
    def calculateDSI(self, sentences): # Code provided from the original DSI paper
        """
        All code in this function is licensed to John D. Patterson from The Pennsylvania State University, 04-04-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
        Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/
        Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
        Please cite Johnson, D. R., Kaufman, J. C., Baker, B. S., Barbot, B., Green, A., van Hell, J., … Beaty, R. (2021, December 1). Extracting Creativity from Narratives using Distributional Semantic Modeling. Retrieved from psyarxiv.com/fmwgy in any publication or presentation

        """
        cos = torch.nn.CosineSimilarity(dim = 0)

        # LOOP OVER SENTENCES AND GET BERT FEATURES (LAYERS 6 & 7)
        features = []  # initialize list to store dcos values, one for each sentence
        words = []
        for i in range(len(sentences)):  # loop over sentences
            sentence = sentences[i].translate(str.maketrans('','',string.punctuation))
            print(f"sentence:{sentence}")
            sent_tokens = self.tokenizer(sentence, max_length = 50, truncation = True, padding = 'max_length', return_tensors="pt")
            sent_words = [self.tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
            sent_indices = np.where(np.in1d(sent_words, self.filter_list, invert = True))[0]  # we'll use this to filter out special tokens and punctuation
            with torch.no_grad():
                sent_output = self.model(**sent_tokens) # feed model the sentence tokens and get outputs
                hids = sent_output.hidden_states # isolate hidden layer activations
            layer6 = hids[6] # isolate layer 6 hidden activations
            layer7 = hids[7] # do the same for layer 7

            for j in sent_indices:  # loop over words and create list of all hidden vectors from layers 6 & 7; additionally store number of words (doubled, to account for layer 6 and 7 duplicates)
                words.append(sent_words[j])
                words.append(sent_words[j])
                features.append(layer6[0,j,:])  # layer 6 features
                features.append(layer7[0,j,:])  # layer 7 features
        
        # GET DCOS VALUES FOR STORY
        num_words = len(words) # number of words, in terms of hidden activation vectors (2*words)
        lower_triangle_indices = np.tril_indices_from(np.random.rand(num_words, num_words), k = -1)  # creates a matrix that represents words*2 (i.e., from word representations from both layer 6+7) and gets the indices of the lower triangle, omitting diagonal (k = -1)A
        story_dcos_vals = []  # intialize storage for dcos of current sentence
        for k in range(len(lower_triangle_indices[0])): # loop over lower triangle indices
            features1 = features[lower_triangle_indices[0][k]]
            features2 = features[lower_triangle_indices[1][k]]
            dcos = (1-cos(features1, features2))  # compute dcos
            story_dcos_vals.append(dcos) # store dcos value in list

        mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()  # get average story dcos
        return mean_story_dcos
    
    def request(self) -> dict:
        DSI_request = Request(
            models=self.models,
            prompt=self.prompts,
            configs=self.configs,
            repeats=len(self.prompts)*self.repeats,
            delay=self.delay
        )

        llm_response = run_request(DSI_request)

        with open(f"responses/{str(self)}.json", "w") as json_file:
            json.dump(llm_response, json_file, indent=4)
            self.return_files.append(f"responses/{str(self)}.json")
        
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
                    if clean_response:
                        llm_response_clean[model][config][idx] = clean_response
                    else:
                        # Couldn't parse response
                        llm_response_clean[model][config][idx] = ""
        
        # Export cleaned responses results
        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            print("Saving clean responses")
            json.dump(llm_response_clean, json_file, indent=4)
        
        return llm_response_clean
    
    def clean_response(self, response):
        if response and isinstance(response, str) and response != "":
            sentences = [s.strip() for s in sent_tokenize(response) if s.strip()]
            return sentences
        return None
    
    def calculate_scores(self, prev: dict | str) -> dict:
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                llm_response = json.load(file)
            self.set_id(prev)

        elif isinstance(prev, dict):
            llm_response = prev
        
        results = {}
        for model, configs in llm_response.items():
            for config, repeats in configs.items():
                model_key = f"{model}_{str(config)}"
                results.setdefault(model_key, {})["results"] = []

                for idx, repeat in enumerate(repeats):
                    if len(repeat) > 0:
                        contains_words = True
                        
                        # Check words appear in the story
                        story_text = keep_letters("".join(repeat).lower())
                        for selected_word in self.selected_words[idx % len(self.selected_words)]:
                            if selected_word.lower().strip() not in story_text: # allows for verbs (i.e. exist) to be conjugated (i.e. existed)
                                contains_words = False
                                break

                        if contains_words:
                            score = self.calculateDSI(repeat)
                            if score is not None:
                                results[model_key]['results'].append(score)
                                print(score)
                        else:
                            results[model_key]['results'].append(0) # if words are not contained in the story append a 0
                    else:
                        print(f"No response was found for one of the responses.")

                results.setdefault(model_key, {})["config"] = json.loads(config)
        
        with open(f"results/{str(self)}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
            self.return_files.append(f"results/{str(self)}.json")
        
        return self.return_files
    
    def __str__(self):
        return "DSI_"+str(self.id)+"_"+str(len(self.models))+"models_"+str(len(self.configs))+"configs"
    
    def run(self):
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        # return self.calculate_scores(prev=prev)
    
    