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
    def __init__(self, models, configs, embedding_models=[GraniteMultilingualEmbeddings], repeats=1, delay=0, n_words=10, languages=["English", "Spanish", "French", "German", "Italian", "Dutch", "Portuguese", "Polish", "Russian", "Japanese", "Hindi"]):
        super().__init__(models=models, configs=configs, embedding_models=embedding_models, repeats=repeats, delay=delay, n_words=n_words, standard_prompt=True, starts_with=None)
        
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
                    f"Reglas: Solo escribe palabras individuales en español. Solo sustantivos (por ejemplo, cosas, objetos, conceptos). No uses nombres propios (por ejemplo, no personas o lugares específicos). "
                    f"Tampoco usa vocabulario especializado (por ejemplo, no términos técnicos). Piensa en las palabras por ti mismo (por ejemplo, no mires solo los objetos a tu alrededor). "
                    f"Haz una lista de estas {str(self.n_words)} palabras, una sola palabra en cada entrada de la lista. Solo escribe las {str(self.n_words)} palabras y nada más." + " " + self.addition_specs
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
            "Japanese": 
                (
                    f"{str(self.n_words)} 個の単語を挙げてください。それぞれの意味や使い方において、できるだけ互いに異なる単語にしてください。 "
                    f"ルール：日本語の単語のみ。名詞のみ（例：物、対象、概念）。固有名詞は禁止（例：特定の人や場所の名前は不可）。 "
                    f"専門的な語彙は禁止（例：技術用語など）。自分で考えてください（例：身の回りの物を見て挙げるだけは不可）。 "
                    f"これらの {str(self.n_words)} 個の単語をリストにし、各項目に1つずつ単語を書いてください。その {str(self.n_words)} 個の単語以外は書かないでください。" + " " + self.addition_specs
                ), 
            "Arabic":
                (
                    f"من فضلك اكتب {str(self.n_words)} كلمة تكون مختلفة قدر الإمكان عن بعضها في جميع المعاني والاستخدامات. "
                    f"القواعد: كلمات مفردة فقط باللغة العربية. أسماء فقط (مثل الأشياء، الأغراض، المفاهيم). "
                    f"بدون أسماء علم (مثل أشخاص أو أماكن محددة). بدون مفردات متخصصة (مثل المصطلحات التقنية). "
                    f"فكّر بالكلمات بنفسك (على سبيل المثال، لا تكتفِ بالنظر إلى الأشياء من حولك). "
                    f"اكتب قائمة بهذه الكلمات وعددها {str(self.n_words)}، كلمة واحدة في كل مدخل. لا تكتب شيئًا آخر غير هذه {str(self.n_words)} كلمة." + " " + self.addition_specs
                ),
            "Chinese":
                (
                    f"请写出 {str(self.n_words)} 个词语，它们在意义和用法上彼此尽可能不同。"
                    f"规则：仅限中文（现代标准汉语书面语）的单个词语。仅限名词（例如：事物、物体、概念）。不允许专有名词（例如：具体的人名或地名）。"
                    f"不使用专业词汇（例如：技术术语）。请自行思考这些词语（不要只是看周围的物品）。"
                    f"将这 {str(self.n_words)} 个词语列成清单，每一项只写一个词语。除了这 {str(self.n_words)} 个词语之外不要写任何其他内容。" + " " + self.addition_specs
                ),
            "Korean":
                (
                    f"{str(self.n_words)}개의 단어를 적어 주세요. 각 단어는 의미와 용법에서 서로 가능한 한 다르게 골라 주세요. "
                    f"규칙: 한국어의 단일 단어만. 명사만(예: 사물, 물체, 개념). 고유명사 금지(예: 특정 인명이나 지명). "
                    f"전문 용어 금지(예: 기술 용어). 스스로 생각해서 적어 주세요(예: 주변에 보이는 것만 적지 마세요). "
                    f"이 {str(self.n_words)}개의 단어를 목록으로 작성하고, 각 항목에는 단어 하나만 적어 주세요. {str(self.n_words)}개의 단어 외에는 아무것도 쓰지 마세요." + " " + self.addition_specs
                ),
            "Czech":
                (
                    f"Napište prosím {str(self.n_words)} slov, která jsou ve svých významech a použitích co nejvíce odlišná. "
                    f"Pravidla: Pouze jednotlivá slova v češtině. Pouze podstatná jména (např. věci, předměty, pojmy). "
                    f"Žádná vlastní jména (např. konkrétní osoby nebo místa). Žádná odborná slovní zásoba (např. technické termíny). "
                    f"Vymyslete slova sami (např. nepište jen to, co vidíte kolem sebe). "
                    f"Vytvořte seznam těchto {str(self.n_words)} slov a do každé položky napište právě jedno slovo. Napište pouze těch {str(self.n_words)} slov a nic dalšího." + " " + self.addition_specs
                ),
        }
        self.languages = list(self.language_prompts.keys())
            
    def __str__(self):
        return "SDAT_"+str(self.id)+"_"+str(len(self.models))+"models_"+str(len(self.configs))+"configs_"+str(self.n_words)+"words"
    
    def request(self) -> dict:
        non_clean_llm_response = []

        for lang in self.languages:
            init_lang = False
            
            HardDAT_request = Request(
                models=self.models,
                prompt=self.language_prompts[lang],
                configs=self.configs,
                repeats=self.repeats,
                delay=self.delay,
                verbose=False
            )
            
            llm_response = run_request(HardDAT_request)
            llm_response["config"] = {}
            llm_response["config"]["language"] = lang

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
            language = entry["config"]["language"]
            repeat = entry["config"]["repeat"]

            for model_key in self.models:
                if model_key in entry:
                    for temp_key, word_lists in entry[model_key].items():
                        assert len(word_lists) == 1 and len(word_lists[0]) == 1
                        merged[model_key][temp_key].extend([self.clean_response(word_lists[0][0])])

        # Optional: Convert defaultdicts to dicts
        final_result = {model: dict(temps) for model, temps in merged.items()}

        with open(f"responses/{str(self)}_clean.json", "w") as json_file:
            json.dump(final_result, json_file, indent=4)

        return final_result

    def run(self):
        prev = self.request()
        prev = self.clean_llm_response(prev=prev)
        return self.calculate_embeddings(prev=prev)
    