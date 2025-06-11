import json
import time
from dataclasses import dataclass, asdict
from typing import List, Union

from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

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

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        return cls(**d)
    
    def __str__(self) -> str:
        non_null_items = (
            f"{k}={v!r}"
            for k, v in asdict(self).items()
            if v is not None
        )
        return '{'+(", ".join(non_null_items))+'}'

class Request():
    def __init__(self,
                 models: Union[str, List[str]], 
                 prompt: str, 
                 configs: Union[dict, List[dict]], 
                 repeats: int = 1, 
                 delay: float = 0.0):
        
        self.models = [models] if isinstance(models, str) else list(models)
        self.prompt = prompt
        self.configs = [configs] if isinstance(configs, dict) else list(configs)
        self.repeats = max(repeats, 1)
        self.delay = max(delay, 0.0)

        print(f"Number of API calls: {len(self.models)*len(self.configs)*self.repeats}")
        print(f"Estimated Time: {(len(self.models)*len(self.configs)*self.repeats*(0.5+self.delay))/60.0}" )

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
        self.delay = request.delay
        self.MODEL_ROUTER = [
            ("gemini-", ChatGoogleGenerativeAI, self._google_kwargs),
            ("google/", ChatVertexAI, self._google_kwargs),
            ("gpt-", ChatOpenAI, self._openai_kwargs),
            ("claude-", ChatAnthropic, self._openai_kwargs)
        ]
        
    def _openai_kwargs(self, gconf: GenerationConfig):
        return {
            "temperature": gconf.temperature,
            "top_p": gconf.top_p,
            "max_tokens": gconf.max_output_tokens,
            "stop": gconf.stop_sequences,
        }

    def _google_kwargs(self, gconf: GenerationConfig):
        return {
            "temperature": gconf.temperature,
            "top_p": gconf.top_p,
            "top_k": gconf.top_k,
            "max_output_tokens": gconf.max_output_tokens,
        }
    
    def build_llm(self, model_name: str, gconf: GenerationConfig):
        for prefix, cls, arg_fn in self.MODEL_ROUTER:
            if model_name.startswith(prefix):
                return cls(model=model_name, **arg_fn(gconf))
        raise ValueError(f"Model name “{model_name}” not matched in MODEL_ROUTER")

    def run(self):
        response = {}

        for model_name, gconf, i in self.request:
            print(f"Requesting {model_name} with config={gconf} ({i + 1})...")
            llm = self.build_llm(model_name, gconf)

            msgs = []
            if gconf.system_instruction:
                msgs.append(SystemMessage(content=gconf.system_instruction))
            msgs.append(HumanMessage(content=self.request.prompt))

            try:
                result = invoke_with_retry(llm, msgs)
                result_content = result.content if hasattr(result, 'content') else result
                response.setdefault(model_name, {}).setdefault(str(gconf), []).append([result_content])
            except Exception as e:
                response.setdefault(model_name, {}).setdefault(str(gconf), []).append(f"Error: {e}")
                print(f"Something went wrong -> Error: {e}")

            time.sleep(self.delay)

        return response

def run_request(request: Request):
    return ModelRunner(request).run()
