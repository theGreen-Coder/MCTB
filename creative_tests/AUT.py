# AUT.py
import re
import json
from typing import List, Callable, Optional
import numpy as np
from embeddings import BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe, SBERT_Encoder
from request import Request, run_request
from scipy.stats import norm
from datetime import datetime
from utils import *
import copy

class SimpleAlternativesUseTask():
    def __init__(
            self,
            models,
            configs,
            target_objects: list = ["brick", "box", "paperclip", "bottle"],
            embedding_models=[SBERT_Encoder],
            repeats: int = 0,
            delay: int = 0,
            n_uses: int = 10,
            standard_prompt: bool = True,
            disallow_common_uses: bool = False,
            score_fn: Optional[Callable] = None,
            file_name=None
        ):
        
        self.models = models
        self.configs = configs
        self.repeats = repeats
        self.embedding_models = embedding_models
        self.target_objects = target_objects
        self.id = str(datetime.now().strftime("%m%d%H%M%S"))
        self.delay = delay
        self.n_uses = n_uses
        self.score_fn = score_fn or calculate_dat_score  # generic diversity scorer
        self.file_name = file_name
        self.standard_prompt = standard_prompt
        self.disallow_common_uses = disallow_common_uses
        self.addition_specs = ''

    def get_AUT_prompt(self, object_name):
        if self.disallow_common_uses:
            self.addition_specs += (
                " Avoid typical or obvious uses; focus on unusual, creative, and feasible uses. Only output the uses, nothing else."
            )

        # Prompts
        if self.standard_prompt:
            return (
                f"List exactly {self.n_uses} alternative uses for a {object_name}. "
                f"Rules: Each entry should be a short phrase (not a full sentence). "
                f"Be specific and concrete; avoid repeating the same category of use. "
                f"Do not add numbering or any extra text—output only the {self.n_uses} uses, one per line."
                + (" " + self.addition_specs if self.addition_specs else "")
            )
        else:
            return [
                (
                    f"Provide {self.n_uses} creative alternative uses for a {object_name.upper()}. "
                    f"Each should be a concise phrase. No explanations, no numbering. Just the {self.n_uses} lines."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"Think of {self.n_uses} non-typical ways to use a {object_name.upper()}. "
                    f"Output only the {self.n_uses} uses, one per line, short phrases only."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"List {self.n_uses} unusual, feasible uses for a {object_name.upper()}. "
                    f"Return strictly {self.n_uses} lines—each line a short phrase describing a distinct use."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"Give {self.n_uses} distinct alternative uses for a {object_name.upper()}. "
                    f"No duplicates or near-duplicates. Output {self.n_uses} short phrases, one per line, and nothing else."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"List exactly {self.n_uses} creative uses for a {object_name.upper()}. "
                    f"Short phrase per line. Do not include numbering or extra commentary."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
            ]
    
    def get_AUT_prompt_judge(self, object_name, use_description):
        return f'You are an expert alternative uses test (AUT) rater. The following is a response of a creative or surprising use of a {object_name.upper()}. On a scale of 10-50, judge how original this use for {object_name.upper()} is, where 10 is "not at all creative" and 50 is "very creative". To rate:"{use_description}. Only output the response and nothing else."'

    def __str__(self):
        if self.file_name is not None and isinstance(self.file_name, str):
            return self.file_name
        safe_obj = re.sub(r"[^A-Za-z0-9]+", "", self.target_objects) or "object"
        return f"AUT_{self.id}_{len(self.models)}models_{len(self.configs)}configs_{self.n_uses}uses_{safe_obj}"

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

    @staticmethod
    def _strip_bullet_or_number(line: str) -> str:
        # Remove leading bullets, dashes, numbers like "1. ", "2) ", "- ", "* ", "— "
        return re.sub(r'^\s*(?:[-*•]+|\d+[\.\)\]:-])\s*', '', line).strip()

    def clean_response(self, response: str) -> List[str] | None:
        """
        Convert raw model text into a list of use phrases.
        Strategy:
          - Prefer newline-separated lists; fallback to comma-separated.
          - Strip bullets/numbering.
          - Keep phrases with 1-12 words; drop empty/very long junk.
        """
        def normalize_and_filter(lines: List[str]) -> List[str]:
            uses = []
            for line in lines:
                line = self._strip_bullet_or_number(line)
                # Remove trailing punctuation that isn't helpful
                line = re.sub(r'[.;:,\s]+$', '', line)
                # Collapse whitespace
                line = re.sub(r'\s+', ' ', line).strip()
                if not line:
                    continue
                word_count = len(line.split())
                if 1 <= word_count <= 12:
                    uses.append(line)
            # Deduplicate while preserving order
            seen = set()
            uniq = []
            for u in uses:
                key = u.lower()
                if key not in seen:
                    seen.add(key)
                    uniq.append(u)
            return uniq

        # Try newline-separated first
        candidates = [s for s in (response or "").split("\n") if s.strip()]
        uses = normalize_and_filter(candidates)
        if len(uses) >= max(7, int(self.n_uses * 0.6)):
            return uses

        # Fallback: comma-separated
        candidates = [s for s in (response or "").split(",") if s.strip()]
        uses = normalize_and_filter(candidates)
        if len(uses) >= max(7, int(self.n_uses * 0.6)):
            return uses

        return None

    def request(self) -> dict:
        non_clean_llm_response = []

        for object_name in self.target_objects:
            AUT_request = Request(
                models=self.models,
                prompt=self.get_AUT_prompt(str(object_name)),
                configs=self.configs,
                repeats=self.repeats,
                default_delay=self.delay,
                verbose=False
            )
            
            llm_response = run_request(AUT_request)
            llm_response["config"] = {}
            llm_response["config"]["object_prompt"] = object_name

            non_clean_llm_response.append(llm_response)
            
            print("Saving to file!")
            with open(f"responses/{str(self)}.json", "w") as json_file:
                json.dump(non_clean_llm_response, json_file, indent=4)
        
        return non_clean_llm_response

    def clean_llm_response(self, prev: dict | str) -> dict:
        # Load from path or use provided dict
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                llm_response = json.load(file)
            self.set_id(prev)
        elif isinstance(prev, dict):
            llm_response = prev
        else:
            raise ValueError("Unsupported type for 'prev'")

        llm_response_clean = copy.deepcopy(llm_response)
        for idx, object_entry in enumerate(llm_response):
            object_name = object_entry["config"]["object_prompt"]
            
            for model_name, val_name in object_entry.items():
                if model_name != "config":
                    for configs, val_lists in val_name.items():
                        for idx2, single_list in enumerate(val_lists):
                            c_response = self.clean_response(single_list[0])
                            llm_response_clean[idx][model_name][configs][idx2] = c_response
        
        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            json.dump(llm_response_clean, json_file, indent=4)
        return llm_response_clean
    
    def calculate_AUT_scores(self, prev):
        return_files = []
        
        # Load cleaned responses
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                llm_response_clean = json.load(file)
            self.set_id(prev)
        elif isinstance(prev, dict):
            llm_response_clean = prev
        else:
            raise ValueError("Unsupported type for 'prev'")
        
        aut_results = copy.deepcopy(llm_response_clean)
        for idx, object_entry in enumerate(llm_response_clean):
            object_name = object_entry["config"]["object_prompt"]
            
            for model_name, val_name in object_entry.items():
                if model_name != "config":
                    for configs, val_lists in val_name.items():
                        for idx2, single_list in enumerate(val_lists):
                            for idx3, use in enumerate(single_list):
                                request_judge = Request(
                                    models="gemini/gemini-2.0-flash",
                                    prompt=self.get_AUT_prompt_judge(str(object_name), use),
                                    configs=[{"temperature": 0.7}],
                                    repeats=1,
                                    default_delay=self.delay,
                                    verbose=False
                                )
                                
                                llm_response = run_request(request_judge)
                                aut_results[idx][model_name][configs][idx2][idx3] = llm_response["gemini/gemini-2.0-flash"]['{"temperature": 0.7}'][0][0]
        
                                with open(f"results/{str(self)}_results.json", "w") as json_file:
                                    json.dump(aut_results, json_file, indent=4)
        return aut_results

    def run(self):
        """
        Full pipeline:
          1. LLM request
          2. Clean responses into lists of uses
          3. Embed + score + normalize
        """
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        return self.calculate_AUT_scores(prev=prev)

