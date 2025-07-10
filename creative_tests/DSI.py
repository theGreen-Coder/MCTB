import re
import json
from typing import List
import numpy as np
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, GloVe, calculate_dat_score
from request import Request, run_request
from scipy.stats import norm
from datetime import datetime

class DivergentAssociationTest():
    def __init__(self, models, configs, embedding_models=[BERT_Encoder_L6, BERT_Encoder_L7], repeats=1, delay=0):
        self.models = models
        self.configs = configs
        self.repeats = repeats
        self.embedding_models = embedding_models
        self.id = str(datetime.now().strftime("%m%d%H%M%S"))
        self.delay = delay
    
    def __str__(self):
        return "DSI_"+str(self.id)+"_"+str(len(self.models))+"models_"+str(len(self.configs))+"configs"
    
    