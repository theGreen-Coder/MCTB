import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = DivergentAssociationTest(
    models=[
            "gemini/gemini-2.5-flash", 
        ],
    configs=[{"temperature": 1, "thinking_budget": 20},
             {"temperature": 1, "thinking_budget": 50},
             {"temperature": 1, "thinking_budget": 100},
             {"temperature": 1, "thinking_budget": 200},
             {"temperature": 1, "thinking_budget": 400},
             {"temperature": 1, "thinking_budget": 300},
             {"temperature": 1, "thinking_budget": 1000},
             {"temperature": 1, "thinking_budget": 1500},
             {"temperature": 1, "thinking_budget": 2000},
             {"temperature": 1, "thinking_budget": 3000},
             {"temperature": 1, "thinking_budget": 4500},
             {"temperature": 1, "thinking_budget": 6000},
             {"temperature": 1, "thinking_budget": 8000},
    ],
    n_words=20,
    repeats=50,
    delay=5,
    file_name="DAT_1.1_effect_thinking_budget"
)

test.run()
