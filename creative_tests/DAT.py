import os
from google import genai
from google.genai import types
from creative_tests.test import CreativeTest

class DivergentAssociationTest(CreativeTest):
    def __init__(self, repetitions, models, embedding_models):
        super().__init__(repetitions, models)
        self.embedding_models = embedding_models

        self.test_prompt = 'Please enter 10 words that are as different from each other as possible, ' \
                            'in all meanings and uses of the words. Rules: Only single words in English. ' \
                            'Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). ' \
                            'No specialized vocabulary (e.g., no technical terms). ' \
                            'Think of the words on your own (e.g., do not just look at objects in your surroundings). ' \
                            'Make a list of these 10 words, a single word in each entry of the list.' \
                            'Do not write anything else, but the 10 words.'
        
    def run(self):
        pass
    

    