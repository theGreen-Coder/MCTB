import os
from google import genai
from google.genai import types
from creative_tests.DivergentTest import DivergentTest

class DivergentAssociationTest(DivergentTest):
    def __init__(self, repetitions=1, models=["google"]):
        super().__init__(repetitions, models)
        self.test_prompt = 'Please enter 10 words that are as different from each other as possible, ' \
                            'in all meanings and uses of the words. Rules: Only single words in English. ' \
                            'Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). ' \
                            'No specialized vocabulary (e.g., no technical terms). ' \
                            'Think of the words on your own (e.g., do not just look at objects in your surroundings). ' \
                            'Make a list of these 10 words, a single word in each entry of the list.'
        
        
    

    