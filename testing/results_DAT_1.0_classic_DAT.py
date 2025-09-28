import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = DivergentAssociationTest(
    models=[
            # "gemma/gemma-3n-e4b-it",
            "gemini/gemini-2.0-flash-lite",
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
        ],
    configs=[
        {"temperature": 1.1},
    ],
    n_words=10,
    repeats=10,
    delay=11,
    file_name="DAT_1.0_classic_DAT"
)

test.run()
