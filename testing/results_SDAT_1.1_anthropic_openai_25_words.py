import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SyntheticDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = SyntheticDivergentAssociationTest(
    models=[
            "claude/claude-sonnet-4-20250514",
            "gpt/gpt-5"
        ],
    configs=[
        {"temperature": 1},
    ],
    n_words=25,
    repeats=15,
    delay=11,
    file_name="SDAT_1.1_anthropic_openai_25_words"
)

test.run()

print("Finished Test!")