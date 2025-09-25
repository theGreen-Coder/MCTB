import json
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Union, Optional

from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions
import openai


@dataclass
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    system_instruction: str | None = None
    stop_sequences: list[str] | None = None
    thinking_budget: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        return cls(**d)
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

class Request():
    def __init__(self,
                 models: Union[str, List[str]], 
                 prompt: Union[str, List[str]], 
                 configs: Union[dict, List[dict]], 
                 repeats: int = 1, 
                 default_delay: float = 0.0,
                 verbose: bool = True,
                 delay_config = "models/time_delay_models_config.json"):
        
        self.models = [models] if isinstance(models, str) else list(models)
        self.prompt = prompt
        self.configs = [configs] if isinstance(configs, dict) else list(configs)
        self.repeats = max(repeats, 1)
        self.default_delay = max(default_delay, 0.0)
        self.delay_config = delay_config
        self._prompt_idx = -1

        if verbose:
            print(f"Number of API calls: {len(self.models)*len(self.configs)*self.repeats}")
            print(f"Estimated Time: {(len(self.models)*len(self.configs)*self.repeats*(0.5+self.default_delay))/60.0} minutes" )

    def set_models(self, models):
        self.models = models
    
    def get_models(self):
        return self.models
    
    def get_prompt(self):
        if isinstance(self.prompt, str):
            return self.prompt

        if not self.prompt:
            raise ValueError("No prompts provided.")

        self._prompt_idx = (self._prompt_idx + 1) % len(self.prompt)
        return self.prompt[self._prompt_idx]
    
    def __iter__(self):
        for model in self.models:
            for conf in self.configs:
                gconf = GenerationConfig.from_dict(conf)
                for i in range(self.repeats):
                    yield model, gconf, i

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=(
        retry_if_exception_type(openai.RateLimitError) |
        retry_if_exception_type(openai.APIError) |
        retry_if_exception_type(openai.Timeout) |
        retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
    ),
    reraise=True
)

def invoke_with_retry(llm, msgs):
    return llm.invoke(msgs)

class ModelRunner():
    def __init__(self, request: Request):
        self.request = request
        self.default_delay = request.default_delay
        
        if isinstance(request.delay_config, str) and request.delay_config != "":
            try:
                with open(request.delay_config, "r") as f:
                    self.delay_config = json.load(f)
            except:
                self.delay_config = None
        
        self.MODEL_ROUTER = {
            "gemini": (ChatGoogleGenerativeAI, self._google_kwargs),
            "google": (ChatVertexAI, self._google_kwargs),
            "gpt": (ChatOpenAI, self._openai_kwargs),
            "claude": (ChatAnthropic, self._anthropic_kwargs),
            "ollama": (ChatOllama, self._ollama_kwargs),
            "custom": (ChatAnthropic, self._custom__kwargs)
        }
    
    def _custom__kwargs(self, gconf: GenerationConfig, model_name: str):
        ### TO BE MODIFIED WITH YOUR CONFIGS
        actual_model = model_name.split("/", 1)[1]
        return {
            "model": actual_model,
            "temperature": gconf.temperature,
            "top_p": gconf.top_p,
            "top_k": gconf.top_k,
            "num_predict": gconf.max_output_tokens,
            "stop": gconf.stop_sequences,
        }

    def _openai_kwargs(self, gconf: GenerationConfig, model_name: Optional[str] = None):
        return {
            "temperature": gconf.temperature,
            "top_p": gconf.top_p,
            "max_tokens": gconf.max_output_tokens,
            "stop": gconf.stop_sequences,
        }

    def _anthropic_kwargs(self, gconf: GenerationConfig, model_name: Optional[str] = None):
        return {
            "temperature": gconf.temperature/2,
            "top_p": gconf.top_p,
            "stop": gconf.stop_sequences,
        }
    
    def _gemma_kwargs(self, gconf: GenerationConfig, model_name: Optional[str] = None):
        return_dict = {
            "temperature": max(gconf.temperature, 1),
            "top_p": gconf.top_p,
            "max_output_tokens": gconf.max_output_tokens,
        }
        return return_dict

    def _google_kwargs(self, gconf: GenerationConfig, model_name: Optional[str] = None):
        return_dict = {
            "temperature": gconf.temperature,
            "top_p": gconf.top_p,
            "top_k": gconf.top_k,
            "max_output_tokens": gconf.max_output_tokens,
            "thinking_budget": gconf.thinking_budget,
        }
        return return_dict
    
    def _ollama_kwargs(self, gconf: GenerationConfig, model_name: Optional[str] = None):
        return_dict = {
            "temperature": gconf.temperature,
            "top_p": gconf.top_p,
            "top_k": gconf.top_k,
            "max_output_tokens": gconf.max_output_tokens,
            "thinking_budget": gconf.thinking_budget,
        }
        return return_dict
    
    def build_llm(self, model_name: str, gconf: GenerationConfig):
        try:
            family, actual_model = model_name.split("/", 1)
        except ValueError:
            raise ValueError(f"Model name must be in format 'family/model', got: {model_name}")

        if family not in self.MODEL_ROUTER:
            raise ValueError(f"Unknown model family: {family}")

        cls, arg_fn = self.MODEL_ROUTER[family]
        return cls(model=actual_model, **arg_fn(gconf, actual_model))


    def run(self):
        response = {}

        for model_name, gconf, i in self.request:
            print(f"Requesting {model_name} with config={gconf} ({i + 1})...")
            llm = self.build_llm(model_name, gconf)

            msgs = []
            if gconf.system_instruction:
                msgs.append(SystemMessage(content=gconf.system_instruction))
            msgs.append(HumanMessage(content=self.request.get_prompt()))

            try:
                print("Waiting for response!")
                result = invoke_with_retry(llm, msgs)
                result_content = result.content if hasattr(result, 'content') else result
                response.setdefault(model_name, {}).setdefault(str(gconf), []).append([result_content])
            except Exception as e:
                print(e)
                response.setdefault(model_name, {}).setdefault(str(gconf), []).append(f"Error: {e}")
                print(f"Something went wrong -> Error: {e}")
            
            cur_delay = self.default_delay
            
            if self.delay_config:
                cur_delay = self.delay_config.get(model_name, self.default_delay)
            
            print(f"Got response! Waiting for {cur_delay} seconds now...")
            time.sleep(cur_delay)
            print()

        return response

def run_request(request: Request):
    return ModelRunner(request).run()
