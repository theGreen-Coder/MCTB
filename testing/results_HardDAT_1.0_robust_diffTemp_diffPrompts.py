import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentAssociationTest, HardDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = HardDivergentAssociationTest(
    models=[
            # "gemma/gemma-3n-e4b-it",
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite"
        ],
    configs=[
        {"temperature": 0.5},
        {"temperature": 1},
        {"temperature": 1.5},
        {"temperature": 2},
    ],
    n_words=25,
    given_words=20,
    common=True,
    delay=11,
    file_name="HardDAT_1.0_robust_diffTemp_diffPrompts"
)

test.run()