import json
import random
import re
from typing import List
from tqdm import tqdm
from embeddings import GraniteMultilingualEmbeddings
from collections import defaultdict
from request import Request, run_request
from creative_tests import DivergentAssociationTest
from utils import *

class SyntheticDivergentAssociationTest(DivergentAssociationTest):
    def __init__(self, models, configs, embedding_models=[GraniteMultilingualEmbeddings], repeats=1, delay=0, n_words=25, languages=["English", "Spanish", "French", "German", "Italian", "Dutch", "Portuguese", "Polish", "Russian", "Japanese", "Hindi"]):
        super().__init__(models=models, configs=configs, embedding_models=embedding_models, repeats=repeats, delay=delay, n_words=n_words, standard_prompt=True, starts_with=None)
        self.languages = languages
        
        self.language_prompts = {
            "English": 
                (
                    f"Please enter {str(self.n_words)} words that are as different from each other as possible, in all meanings and uses of the words. "
                    f"Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). "
                    f"No specialized vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). "
                    f"Make a list of these {str(self.n_words)} words, a single word in each entry of the list. Do not write anything else but the {str(self.n_words)} words." + " " + self.addition_specs
                ), 
            "Spanish": 
                (
                    f"Por favor, escribe {str(self.n_words)} palabras que sean lo más diferentes posible entre sí, en todos sus significados y usos. "
                    f"Reglas: Solo palabras individuales en español. Solo sustantivos (por ejemplo, cosas, objetos, conceptos). No nombres propios (por ejemplo, no personas o lugares específicos). "
                    f"No vocabulario especializado (por ejemplo, no términos técnicos). Piensa en las palabras por ti mismo (por ejemplo, no mires solo los objetos a tu alrededor). "
                    f"Haz una lista de estas {str(self.n_words)} palabras, una sola palabra en cada entrada de la lista. No escribas nada más que las {str(self.n_words)} palabras." + " " + self.addition_specs
                ), 
            "French": 
                (
                    f"Veuillez écrire {str(self.n_words)} mots qui soient aussi différents les uns des autres que possible, dans tous leurs sens et usages. "
                    f"Règles : Uniquement des mots simples en français. Uniquement des noms (par exemple, choses, objets, concepts). Pas de noms propres (par exemple, pas de personnes ou de lieux spécifiques). "
                    f"Pas de vocabulaire spécialisé (par exemple, pas de termes techniques). Trouvez les mots par vous-même (par exemple, ne vous contentez pas de regarder les objets autour de vous). "
                    f"Faites une liste de ces {str(self.n_words)} mots, un mot par entrée. N’écrivez rien d’autre que ces {str(self.n_words)} mots." + " " + self.addition_specs
                ), 
            "German": 
                (
                    f"Bitte geben Sie {str(self.n_words)} Wörter ein, die sich in allen Bedeutungen und Verwendungen so stark wie möglich voneinander unterscheiden. "
                    f"Regeln: Nur einzelne Wörter auf Deutsch. Nur Substantive (z. B. Dinge, Objekte, Konzepte). Keine Eigennamen (z. B. keine bestimmten Personen oder Orte). "
                    f"Kein Fachvokabular (z. B. keine Fachbegriffe). Denken Sie sich die Wörter selbst aus (z. B. nicht nur Gegenstände in Ihrer Umgebung betrachten). "
                    f"Erstellen Sie eine Liste dieser {str(self.n_words)} Wörter, jeweils ein einzelnes Wort pro Eintrag. Schreiben Sie nichts anderes als die {str(self.n_words)} Wörter." + " " + self.addition_specs
                ), 
            "Italian": 
                (
                    f"Per favore, inserisci {str(self.n_words)} parole che siano il più diverse possibile tra loro, in tutti i significati e usi. "
                    f"Regole: Solo parole singole in italiano. Solo sostantivi (ad esempio, cose, oggetti, concetti). Niente nomi propri (ad esempio, nessuna persona o luogo specifico). "
                    f"Nessun vocabolario specializzato (ad esempio, nessun termine tecnico). Pensa alle parole da solo (ad esempio, non limitarti a guardare gli oggetti intorno a te). "
                    f"Fai un elenco di queste {str(self.n_words)} parole, una sola parola per ogni voce. Non scrivere altro che le {str(self.n_words)} parole." + " " + self.addition_specs
                ), 
            "Dutch": 
                (
                    f"Voer alstublieft {str(self.n_words)} woorden in die zo verschillend mogelijk van elkaar zijn, in alle betekenissen en gebruikswijzen. "
                    f"Regels: Alleen losse woorden in het Nederlands. Alleen zelfstandige naamwoorden (bijvoorbeeld dingen, objecten, concepten). Geen eigennamen (bijvoorbeeld geen specifieke mensen of plaatsen). "
                    f"Geen gespecialiseerd vocabulaire (bijvoorbeeld geen technische termen). Bedenk de woorden zelf (bijvoorbeeld niet alleen door naar objecten in uw omgeving te kijken). "
                    f"Maak een lijst van deze {str(self.n_words)} woorden, één woord per item. Schrijf niets anders dan de {str(self.n_words)} woorden." + " " + self.addition_specs
                ), 
            "Portuguese": 
                (
                    f"Por favor, escreva {str(self.n_words)} palavras que sejam o mais diferentes possível entre si, em todos os significados e usos. "
                    f"Regras: Apenas palavras únicas em português. Apenas substantivos (por exemplo, coisas, objetos, conceitos). Nenhum nome próprio (por exemplo, nenhuma pessoa ou lugar específico). "
                    f"Nenhum vocabulário especializado (por exemplo, nenhum termo técnico). Pense nas palavras por conta própria (por exemplo, não apenas olhando para os objetos ao seu redor). "
                    f"Faça uma lista dessas {str(self.n_words)} palavras, uma palavra por item. Não escreva nada além das {str(self.n_words)} palavras." + " " + self.addition_specs
                ), 
            "Polish": 
                (
                    f"Proszę podać {str(self.n_words)} słów, które będą jak najbardziej różne od siebie, we wszystkich znaczeniach i zastosowaniach. "
                    f"Zasady: Tylko pojedyncze słowa po polsku. Tylko rzeczowniki (np. rzeczy, obiekty, pojęcia). Bez nazw własnych (np. żadnych konkretnych osób ani miejsc). "
                    f"Bez specjalistycznego słownictwa (np. terminów technicznych). Wymyśl słowa samodzielnie (np. nie patrz tylko na przedmioty wokół siebie). "
                    f"Sporządź listę tych {str(self.n_words)} słów, jedno słowo w każdym wpisie. Nie pisz nic oprócz tych {str(self.n_words)} słów." + " " + self.addition_specs
                ), 
            "Russian": 
                (
                    f"Пожалуйста, введите {str(self.n_words)} слов, которые как можно больше отличаются друг от друга, во всех значениях и употреблениях. "
                    f"Правила: Только отдельные слова на русском языке. Только существительные (например, вещи, предметы, понятия). Без собственных имён (например, без конкретных людей или мест). "
                    f"Без специализированной лексики (например, без технических терминов). Придумайте слова самостоятельно (например, не ограничивайтесь предметами вокруг себя). "
                    f"Составьте список из этих {str(self.n_words)} слов, по одному слову в каждой записи. Не пишите ничего, кроме этих {str(self.n_words)} слов." + " " + self.addition_specs
                ), 
            "Japanese": 
                (
                    f"{str(self.n_words)} 個の単語を挙げてください。それぞれの意味や使い方において、できるだけ互いに異なる単語にしてください。 "
                    f"ルール：日本語の単語のみ。名詞のみ（例：物、対象、概念）。固有名詞は禁止（例：特定の人や場所の名前は不可）。 "
                    f"専門的な語彙は禁止（例：技術用語など）。自分で考えてください（例：身の回りの物を見て挙げるだけは不可）。 "
                    f"これらの {str(self.n_words)} 個の単語をリストにし、各項目に1つずつ単語を書いてください。その {str(self.n_words)} 個の単語以外は書かないでください。" + " " + self.addition_specs
                ), 
            "Hindi": 
                (
                    f"कृपया {str(self.n_words)} शब्द लिखें जो एक-दूसरे से अर्थ और प्रयोग में जितने अलग हो सकें उतने अलग हों। "
                    f"नियम: केवल एकल शब्द हिन्दी में। केवल संज्ञाएँ (जैसे वस्तुएँ, चीजें, अवधारणाएँ)। कोई व्यक्तिवाचक संज्ञा नहीं (जैसे विशेष व्यक्ति या स्थान नहीं)। "
                    f"कोई तकनीकी या विशेष शब्दावली नहीं। शब्द स्वयं सोचें (जैसे केवल आसपास की वस्तुओं को देखकर न लिखें)। "
                    f"इन {str(self.n_words)} शब्दों की सूची बनाएँ, सूची के प्रत्येक प्रविष्टि में केवल एक शब्द लिखें। {str(self.n_words)} शब्दों के अलावा कुछ और न लिखें।" + " " + self.addition_specs
                ),
        }
            
    def __str__(self):
        return "SDAT_"+str(self.id)+"_"+str(len(self.models))+"models_"+str(len(self.configs))+"configs_"+str(self.n_words)+"words"
    
    def request(self) -> dict:
        non_clean_llm_response = []

        for lang in self.languages:
            for repeat in range(self.repeats):
                HardDAT_request = Request(
                    models=self.models,
                    prompt=self.language_prompts[lang],
                    configs=self.configs,
                    repeats=1,
                    delay=self.delay,
                    verbose=False
                )

                llm_response = run_request(HardDAT_request)
                llm_response["config"] = {}
                llm_response["config"]["language"] = lang
                llm_response["config"]["repeat"] = repeat

                non_clean_llm_response.append(llm_response)
        
        with open(f"responses/{str(self)}.json", "w") as json_file:
            json.dump(non_clean_llm_response, json_file, indent=4)
        
        return non_clean_llm_response

    def clean_llm_response(self, prev: dict | str) -> dict:
        # Check input
        if isinstance(prev, str):
            with open(prev, 'r') as file:
                non_clean_llm_response = json.load(file)
            self.set_id(prev)

        elif isinstance(prev, dict) or isinstance(prev, List):
            non_clean_llm_response = prev
    
        merged = defaultdict(lambda: defaultdict(list))

        for entry in non_clean_llm_response:
            random_words = entry["config"]["given_words"]
            language = entry["config"]["language"]
            repeat = entry["config"]["repeat"]

            for model_key in self.models:
                if model_key in entry:
                    for temp_key, word_lists in entry[model_key].items():
                        # all_words = [clean_response(word) for word in word_lists]
                        # all_words += random_words
                        assert len(word_lists) == 1 and len(word_lists[0]) == 1
                        merged[model_key][temp_key].extend([self.clean_response(word_lists[0][0]) + random_words])

        # Optional: Convert defaultdicts to dicts
        final_result = {model: dict(temps) for model, temps in merged.items()}

        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            json.dump(final_result, json_file, indent=4)

        return final_result

    def run(self):
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        return self.calculate_embeddings(prev=prev)
    