import os
import json
import numpy as np
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import GloVe
from request import Request, run_request
from scipy.stats import norm
from datetime import datetime
from creative_tests.DAT import DivergentAssociationTest
from utils import *

test = DivergentAssociationTest(
    models=[
        "gemma-3-27b-it",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.0-flash", 
        "gemini-1.5-flash",
        ],
    configs=[{"temperature": 0.5} ,{"temperature": 1}, {"temperature": 1.5}, {"temperature": 2}],
    repeats=20,
    delay=6,
)

test.run()