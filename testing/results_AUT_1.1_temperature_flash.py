import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SimpleAlternativesUseTask
from utils import generate_configs
from dotenv import load_dotenv

load_dotenv()

test = SimpleAlternativesUseTask(
    models=[
            "gemini/gemini-2.5-flash-lite",
        ],
    configs=generate_configs(start = 0.5, end = 2.0, step = 0.15),
    target_objects = ["brick", "paperclip"],
    n_uses = 20,
    repeats = 50,
    standard_prompt = False,
    disallow_common_uses = True,
    file_name="AUT_1.1_temperature_flash",
    delay = 1,
)

test.request()

print("Finished Test!")