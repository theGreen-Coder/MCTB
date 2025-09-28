import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SimpleAlternativesUseTask
from dotenv import load_dotenv

load_dotenv()

test = SimpleAlternativesUseTask(
    models=[
            "gemini/gemini-2.5-pro", 
        ],
    configs=[
        {"temperature": 0.5},
        {"temperature": 1.0},
        {"temperature": 1.5},
    ],
    target_objects = ["brick", "box", "paperclip", "bottle", "rope", "book", "table", "shovel"],
    n_uses = 20,
    repeats = 15,
    standard_prompt = False,
    disallow_common_uses = True,
    file_name="AUT_1.3_gemini2.5-pro",
    delay = 1,
)

test.request()

print("Finished Test!")