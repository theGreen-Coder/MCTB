import os
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import DivergentSemanticIntegration
from dotenv import load_dotenv

load_dotenv()

test = DivergentSemanticIntegration(
    models=[
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "gemma/gemma-3n-e4b-it"],
    configs=[
        {"temperature": 1},
    ],
    variant_prompts=[
        "Write a short creative story consisting of exactly five sentences using the three-word prompt: ||words||. Make sure to use all three words in the story. Only output the story, nothing else.",
        "Please craft a five-sentence imaginative story that includes the following words: ||words||. Use each word at least once and keep the writing creative. Provide only the story, no extra commentary.",
        "Write a creative story of five sentences based on this three-word prompt: ||words||. Be sure all three words appear in your story. Share only the story itself, without any additional explanation.",
        "Using the prompt ||words||, create a short story that is exactly five sentences long. Every word must be included, and the story should be imaginative. Output just the story and nothing more.",
        "Please generate a five-sentence story inspired by the following three words: ||words||. Ensure all three words are used. Do not add anything besides the story itself."
    ],
    repeats=1, # 3 * 8 = 24 repeats
    delay=5,
)

test.run()