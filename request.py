import os
from google import genai
from google.genai import types
from google.api_core import retry

class RequestResponse():
    def __init__(self, model, prompt, config):
        self.model = model
        self.prompt = prompt
        self.config = config
    
    def request(self):
        pass
    
class RequestResponseGoogle(RequestResponse):
    def __init__(self, model, prompt, config):
        super().__init__(model, prompt, config)
    
    def request(self):
        is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        genai.models.Models.generate_content = retry.Retry( predicate=is_retriable)(genai.models.Models.generate_content)

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        google_config = types.GenerateContentConfig(
            temperature        = self.config.get("temperature"),
            top_p              = self.config.get("top_p", ),
            top_k              = self.config.get("top_k"),
            max_output_tokens  = self.config.get("max_output_tokens"),
            system_instruction = self.config.get("system_instruction"),
            stop_sequences     = self.config.get("stop_sequences")
        )

        response = client.models.generate_content(
            model=self.model,
            config=google_config,
            contents=self.prompt
        )

        return response