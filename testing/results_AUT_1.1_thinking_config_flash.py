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
    target_objects = ["brick", "paperclip"],
    n_uses = 20,
    repeats = 50,
    standard_prompt = False,
    disallow_common_uses = True,
    file_name="AUT_1.0_thinking_config_flash",
    delay = 1,
)

test.request()

print("Finished Test!")