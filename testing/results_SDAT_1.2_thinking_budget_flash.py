import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SyntheticDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = SyntheticDivergentAssociationTest(
    models=["gemini/gemini-2.5-flash-lite"],
    configs = [
                {"temperature": 1, "thinking_budget": 512},
                {"temperature": 1, "thinking_budget": 768},
                {"temperature": 1, "thinking_budget": 1024},
                {"temperature": 1, "thinking_budget": 1536},
                {"temperature": 1, "thinking_budget": 2048},
                {"temperature": 1, "thinking_budget": 3072},
                {"temperature": 1, "thinking_budget": 4096},
                {"temperature": 1, "thinking_budget": 6144},
                {"temperature": 1, "thinking_budget": 8192},
                {"temperature": 1, "thinking_budget": 12288},
                {"temperature": 1, "thinking_budget": 24000},
    ],
    n_words=10,
    repeats=15,
    delay=11,
    file_name="SDAT_1.2_thinking_budget_flash"
)

test.run()

print("Finished Test!")