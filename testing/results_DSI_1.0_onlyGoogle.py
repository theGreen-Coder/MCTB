import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentSemanticIntegration
from dotenv import load_dotenv

load_dotenv()

test = DivergentSemanticIntegration(
    models=["gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "gemma/gemma-3n-e4b-it"
        ],
    configs=[
        {"temperature": 1},
    ],
    repeats=3, # 3 * 8 = 24 repeats
    delay=15,
)

test.run()