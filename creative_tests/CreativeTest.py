import abc

class CreativeTest(abc.ABC):
    def __init__(self, repetitions, models):
        self.repetitions = repetitions
        self.models = models
        self.test_prompt = ''
        self.results = {}
    
    def get_test_prompt(self):
        return self.test_prompt
    
    def set_test_prompt(self, test_prompt):
        self.test_prompt = test_prompt
    
    def test(self):
        pass

    def get_statistics(self):
        pass