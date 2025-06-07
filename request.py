import os
import abc
from google import genai
from google.genai import types
from typing import List, Union
from google.api_core import retry
from dataclasses import dataclass
import time
import json

@dataclass
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    system_instruction: str | None = None
    stop_sequences: list[str] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        return cls(**d)

class Request():
    def __init__(self,
                 models: Union[str, List[str]], 
                 prompt: str, 
                 configs: Union[dict, List[dict]], 
                 repeats: int = 1):
        
        self.models = [models] if isinstance(models, str) else list(models)
        self.prompt = prompt
        self.configs = [configs] if isinstance(configs, dict) else list(configs)
        self.repeats = max(repeats, 1)

    def set_models(self, models):
        self.models = models
    
    def get_models(self):
        return self.models
    
    def __iter__(self):
        for model in self.models:
            for conf in self.configs:
                gconf = GenerationConfig.from_dict(conf)
                for i in range(self.repeats):
                    yield model, gconf, i

class BaseRunner(abc.ABC):
    def __init__(self, request: Request, delay: float = 0.0) -> None:
        self.request = request
        self.delay = delay
    
    @abc.abstractmethod
    def run(self): """Must implement method in all Runner classes."""
    
class RunnerGoogle(BaseRunner):
    def __init__(self, request: Request, delay: float = 0.0) -> None:
        super().__init__(request, delay)
    
    def run(self):
        response = {}

        is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
        genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        
        for model, gconf, i in self.request:
            google_config = types.GenerateContentConfig(
                temperature        = gconf.temperature,
                top_p              = gconf.top_p,
                top_k              = gconf.top_k,
                max_output_tokens  = gconf.max_output_tokens,
                system_instruction = gconf.system_instruction,
                stop_sequences     = gconf.stop_sequences
            )

            # Make API call
            model_response = client.models.generate_content(
                model=model,
                config=google_config,
                contents=self.request.prompt
            )

            # Possibly update this to handle Google specific response format
            response_format = {
                "config": gconf,
                "response": model_response,
            }

            response.setdefault(model, []).append(response_format)

            time.sleep(self.delay)
        
        return response

def run_request(request: Request, delay=0):
    response = {}

    with open("models.json") as f:
        data = json.load(f)

    google_models = [m for models in data["google"].values() for m in models]
    openai_models = [m for models in data["openai"].values() for m in models]

    models = {}
    
    for model in request.get_models():
        print(model)
        match model:
            case model if model in google_models:
                models['google'] = models.get('google', []) + [model]
            case model if model in openai_models:
                models['openai'] = models.get('openai', []) + [model]
            case _:
                print("Model not recognized!")
    
    for model_company, model_list in models.items():
        match model_company:
            case 'google':
                google_request = request
                google_request.set_models(model_list)

                google_runner = RunnerGoogle(google_request, delay=delay)
                print("HELLOO!!")
                response['google'] = google_runner.run()
            
            case 'openai':
                # TO implement
                pass
    
    return response
