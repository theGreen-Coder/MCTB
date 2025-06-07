import os
from google import genai
from google.genai import types
from google.api_core import retry
from request import Request, run_request

test_request = Request(
    models="gemini-2.0-flash", 
    prompt="Write hello world in python", 
    configs={
        "temperature": 0.5,
    },
    repeats=1,
)

result = run_request(test_request, delay=0)