import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SyntheticDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = SyntheticDivergentAssociationTest(
    models=[
            "ollama/gemma3n:e4b",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
        ],
    configs=[
        {"temperature": 1},
    ],
    n_words=25,
    repeats=25,
    delay=11,
    file_name="SDAT_1.0_onlyGoogle_25_words"
)

test.run()

print("Finished Test!")
