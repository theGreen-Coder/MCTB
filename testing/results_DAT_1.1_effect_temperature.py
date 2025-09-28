import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

def generate_configs(start, end, step):
    configs = []
    temperature = start
    while temperature <= end + 1e-9:  # small tolerance for floating point issues
        configs.append({"temperature": round(temperature, 2)})
        temperature += step
    return configs

print(generate_configs(start = 0.5, end = 2.0, step = 0.15))

test = DivergentAssociationTest(
    models=[
            "gemini/gemini-2.5-flash", 
        ],
    configs=generate_configs(start = 0.5, end = 2.0, step = 0.15),
    n_words=20,
    repeats=50,
    delay=5,
    file_name="DAT_1.1_effect_temperature"
)

test.run()
