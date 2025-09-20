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


class SimpleAlternativesUseTask():
    def __init__(
            self,
            models,
            configs,
            target_object: str = "brick",
            embedding_models=[SBERT_Encoder],
            repeats: int = 0,
            delay: int = 0,
            n_uses: int = 10,
            standard_prompt: bool = True,
            disallow_common_uses: bool = False,
            score_fn: Optional[Callable] = None
        ):
        
        self.models = models
        self.configs = configs
        self.repeats = repeats
        self.embedding_models = embedding_models
        self.target_object = target_object
        self.id = str(datetime.now().strftime("%m%d%H%M%S"))
        self.delay = delay
        self.n_uses = n_uses
        self.score_fn = score_fn or calculate_dat_score  # generic diversity scorer

        # Additional constraints for the instruction
        self.addition_specs = ''
        if disallow_common_uses:
            self.addition_specs += (
                " Avoid typical or obvious uses; focus on unusual, creative, and feasible uses."
            )

        # Prompts
        if standard_prompt:
            self.test_prompt = (
                f"List exactly {self.n_uses} alternative uses for a {self.target_object}. "
                f"Rules: Each entry should be a short phrase (not a full sentence). "
                f"Be specific and concrete; avoid repeating the same category of use. "
                f"Do not add numbering or any extra text—output only the {self.n_uses} uses, one per line."
                + (" " + self.addition_specs if self.addition_specs else "")
            )
        else:
            self.test_prompt = [
                (
                    f"Provide {self.n_uses} creative alternative uses for a {self.target_object}. "
                    f"Each should be a concise phrase. No explanations, no numbering—just the {self.n_uses} lines."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"Think of {self.n_uses} non-typical ways to use a {self.target_object}. "
                    f"Output only the {self.n_uses} uses, one per line, short phrases only."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"List {self.n_uses} unusual, feasible uses for a {self.target_object}. "
                    f"Return strictly {self.n_uses} lines—each line a short phrase describing a distinct use."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"Give {self.n_uses} distinct alternative uses for a {self.target_object}. "
                    f"No duplicates or near-duplicates. Output {self.n_uses} short phrases, one per line, and nothing else."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
                (
                    f"List exactly {self.n_uses} creative uses for a {self.target_object}. "
                    f"Short phrase per line. Do not include numbering or extra commentary."
                    + (" " + self.addition_specs if self.addition_specs else "")
                ),
            ]

    def __str__(self):
        safe_obj = re.sub(r"[^A-Za-z0-9]+", "", self.target_object) or "object"
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
          - Keep phrases with 1–12 words; drop empty/very long junk.
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
        aut_request = Request(
            models=self.models,
            prompt=self.test_prompt,
            configs=self.configs,
            repeats=self.repeats,
            delay=self.delay
        )
        llm_response = run_request(aut_request)
        with open(f"responses/{str(self)}.json", "w") as json_file:
            json.dump(llm_response, json_file, indent=4)
        return llm_response

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

        llm_response_clean = llm_response
        for model, configs in llm_response.items():
            for config, repeats in configs.items():
                for idx, repeat in enumerate(repeats):
                    response = repeat[0] if isinstance(repeat, list) and len(repeat) > 0 else ""
                    cleaned = self.clean_response(response=response) or ""
                    llm_response_clean[model][config][idx] = cleaned

        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            json.dump(llm_response_clean, json_file, indent=4)
        return llm_response_clean

    def calculate_embeddings(self, prev) -> List[str]:
        """
        For each model/config and each embedding model:
          - Compute a diversity score list (one per repeat) using `self.score_fn`
          - Save unnormalized + normalized results
        Returns a list of result file paths.
        """
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

        results = {}
        model_distribution = {}

        self.init_word_embeddings()

        for model, configs in llm_response_clean.items():
            for config, repeats in configs.items():
                model_key = f"{model}_{str(config)}"

                for embedding_model in self.embedding_models:
                    emb_key = str(embedding_model)
                    scores = []
                    originality_scores = []
                    flexibility_scores = []

                    for repeat_idx, repeat in enumerate(repeats):
                        if isinstance(repeat, list) and len(repeat) > 0:
                            try:
                                # === 1. Diversity score (your existing metric) ===
                                score = self.score_fn(embedding_model, repeat)
                                if score is not None:
                                    scores.append(float(score))

                                # === 2. Originality (row-wise: prompt vs each response) ===
                                for response in repeat:
                                    try:
                                        prompt_vec = embedding_model(text=self.test_prompt)
                                        resp_vec = embedding_model(text=response)
                                        cos_score = cos_sim(prompt_vec, resp_vec).item()
                                        originality_scores.append(1 - cos_score)
                                    except Exception as e:
                                        print(f"Originality failed {model_key}/{emb_key}: {e}")

                                # === 3. Flexibility (group-wise: adjacent responses) ===
                                try:
                                    resp_vecs = [embedding_model(text=r) for r in repeat]
                                    flex_score = 0
                                    for i in range(len(resp_vecs) - 1):
                                        flex_score += 1 - cos_sim(resp_vecs[i], resp_vecs[i + 1]).item()
                                    flexibility_scores.append(flex_score)
                                except Exception as e:
                                    print(f"Flexibility failed {model_key}/{emb_key}: {e}")

                            except Exception as e:
                                print(f"Scoring failed for {model_key} / {emb_key}: {e}")
                        else:
                            print("Empty or invalid repeat encountered; skipping.")

                    # Store everything
                    results.setdefault(model_key, {})[emb_key] = {
                        "diversity": scores,
                        "originality": originality_scores,
                        "flexibility": flexibility_scores,
                    }
                    model_distribution.setdefault(emb_key, []).extend(scores)

                # Persist the decoded config alongside the scores (like in DAT)
                try:
                    results.setdefault(model_key, {})["config"] = json.loads(config)
                except Exception:
                    results.setdefault(model_key, {})["config"] = config


        # Save unnormalized
        with open(f"results/{str(self)}_unnormalized.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
        return_files.append(f"results/{str(self)}_unnormalized.json")

        # Normalize per embedding model to 0–100 via CDF of z-scores
        stats = {}
        for emb_key, all_scores in model_distribution.items():
            a = np.asarray(all_scores, dtype=float) if len(all_scores) > 0 else np.asarray([0.0])
            mean, std = a.mean(), a.std()
            stats[emb_key] = (mean, std if std > 0 else None)

        normalized_results = {}
        for model_key, emb_dict in results.items():
            normalized_results[model_key] = {}
            for emb_key, raw_scores in emb_dict.items():
                if emb_key == "config":
                    normalized_results[model_key]["config"] = raw_scores
                    continue

                mean, std = stats[emb_key]
                if std is None:
                    normed = [50.0] * len(raw_scores)
                else:
                    normed = [float(norm.cdf((s - mean) / std) * 100.0) for s in raw_scores]
                normalized_results[model_key][emb_key] = normed

        with open(f"results/{str(self)}_normalized.json", "w") as json_file:
            json.dump(normalized_results, json_file, indent=4)
        return_files.append(f"results/{str(self)}_normalized.json")

        return return_files

    def run(self):
        """
        Full pipeline:
          1. LLM request
          2. Clean responses into lists of uses
          3. Embed + score + normalize
        """
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        return self.calculate_embeddings(prev=prev)

