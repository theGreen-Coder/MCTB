import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SimpleAlternativesUseTask
from dotenv import load_dotenv

load_dotenv()

test = SimpleAlternativesUseTask(
    models=[
            "ollama/gemma3n:e4b",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "claude/claude-sonnet-4-20250514",
            "gpt/gpt-5"
        ],
    configs=[
        {"temperature": 0.7},
    ],
    target_objects = ["brick", "box", "paperclip", "bottle"],
    n_uses = 20,
    repeats = 25,
    standard_prompt = False,
    disallow_common_uses = True,
    file_name="AUT_1.0_all_models",
    delay = 1,
)

test.request()

print("Finished Test!")