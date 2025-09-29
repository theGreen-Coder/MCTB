from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

test = DivergentAssociationTest(
    models=[
            "claude/claude-sonnet-4-20250514",
            "gpt/gpt-5"
        ],
    configs=[
        {"temperature": 0.5},
        {"temperature": 1},
        {"temperature": 1.5},
        {"temperature": 2},
    ],
    n_words=20,
    repeats=20,
    delay=11,
    standard_prompt=False,
    file_name="DAT_1.3_claude_openai"
)

test.run()