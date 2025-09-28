import os
import time
from embeddings import BERT_Encoder_L6, BERT_Encoder_L7, BERT_WordEmbeddings_L6, BERT_WordEmbeddings_L7, GloVe
from creative_tests import SyntheticDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

language_constraints_edible = {
    "English": "Final rule: every word must be something you can eat.", 
    "Spanish": "Regla final: cada palabra debe ser algo que puedas comer.", 
    "French": "Règle finale : chaque mot doit être quelque chose que vous pouvez manger.", 
    "German": "Endregel: Jedes Wort muss etwas sein, das man essen kann.", 
    "Italian": "Regola finale: ogni parola deve essere qualcosa che si può mangiare.", 
    "Dutch": "Laatste regel: elk woord moet iets zijn dat je kunt eten.", 
    "Portuguese": "Regra final: cada palavra deve ser algo que você possa comer.", 
    "Japanese": "最終ルール：すべての単語は食べられるものでなければなりません。", 
    "Arabic": "القاعدة النهائية: يجب أن تكون كل كلمة شيئًا يمكنك أكله.", 
    "Chinese": "最终规则：每个词都必须是可以吃的东西。", 
    "Korean": "최종 규칙: 모든 단어는 먹을 수 있는 것이어야 합니다.", 
    "Czech": "Konečné pravidlo: každé slovo musí být něco, co se dá sníst.", 
}

language_constraints_tools = {
    "English": "Final rule: every word must be a tool.", 
    "Spanish": "Regla final: cada palabra debe ser una herramienta.", 
    "French": "Règle finale : chaque mot doit être un outil.", 
    "German": "Endregel: Jedes Wort muss ein Werkzeug sein.", 
    "Italian": "Regola finale: ogni parola deve essere uno strumento.", 
    "Dutch": "Laatste regel: elk woord moet een gereedschap zijn.", 
    "Portuguese": "Regra final: cada palavra deve ser uma ferramenta.", 
    "Japanese": "最終ルール：すべての単語は道具でなければなりません。", 
    "Arabic": "القاعدة النهائية: يجب أن تكون كل كلمة أداة.", 
    "Chinese": "最终规则：每个词都必须是工具。", 
    "Korean": "최종 규칙: 모든 단어는 도구여야 합니다.", 
    "Czech": "Konečné pravidlo: každé slovo musí být nástroj.", 
}

language_constraints_money = {
    "English": "Final rule: each word must be represented by a physical object that can be acquired for under $10.", 
    "Spanish": "Regla final: cada palabra debe estar representada por un objeto físico que se pueda adquirir por menos de 10 dólares.", 
    "French": "Règle finale : chaque mot doit être représenté par un objet physique pouvant être acquis pour moins de 10 dollars.", 
    "German": "Endregel: Jedes Wort muss durch einen physischen Gegenstand dargestellt werden, der für weniger als 10 Dollar erhältlich ist.", 
    "Italian": "Regola finale: ogni parola deve essere rappresentata da un oggetto fisico che si può acquistare per meno di 10 dollari.", 
    "Dutch": "Laatste regel: elk woord moet worden vertegenwoordigd door een fysiek object dat voor minder dan 10 dollar te verkrijgen is.", 
    "Portuguese": "Regra final: cada palavra deve ser representada por um objeto físico que possa ser adquirido por menos de 10 dólares.", 
    "Japanese": "最終ルール：すべての単語は10ドル未満で手に入る物理的なものでなければなりません。", 
    "Arabic": "القاعدة النهائية: يجب أن تمثل كل كلمة شيئًا ماديًا يمكن الحصول عليه بأقل من 10 دولارات.", 
    "Chinese": "最终规则：每个词都必须是可以用不到10美元获得的实物。", 
    "Korean": "최종 규칙: 모든 단어는 10달러 이하로 구입할 수 있는 물건이어야 합니다.", 
    "Czech": "Konečné pravidlo: každé slovo musí být fyzický předmět, který lze získat za méně než 10 dolarů.", 
}

language_constraints_sound = {
    "English": "Final rule: each word must be an item/object/phenomena that makes a sound when used.",
    "Spanish": "Regla final: cada palabra debe ser un ítem/objeto/fenómeno que haga un sonido cuando se use.",
    "French": "Règle finale : chaque mot doit être un élément/objet/phénomène qui produit un son lorsqu’il est utilisé.",
    "German": "Endregel: Jedes Wort muss ein Element/Objekt/Phänomen sein, das bei Verwendung ein Geräusch erzeugt.",
    "Italian": "Regola finale: ogni parola deve essere un elemento/oggetto/fenomeno che produce un suono quando viene utilizzato.",
    "Dutch": "Laatste regel: elk woord moet een item/voorwerp/fenomeen zijn dat een geluid maakt wanneer het wordt gebruikt.",
    "Portuguese": "Regra final: cada palavra deve ser um item/objeto/fenômeno que emite um som quando for usado.",
    "Japanese": "最終ルール：各単語は、使用時に音を発するアイテム／物体／現象でなければならない。",
    "Arabic": "القاعدة النهائية: يجب أن تكون كل كلمة عنصرًا/شيئًا/ظاهرة تُصدر صوتًا عند استخدامها.",
    "Chinese": "最终规则：每个词都必须是使用时会发出声音的物品/对象/现象。",
    "Korean": "최종 규칙: 모든 단어는 사용 시 소리를 내는 항목/물체/현상이어야 합니다.",
    "Czech": "Konečné pravidlo: každé slovo musí být položka/předmět/fenomén, který při použití vydává zvuk.",
}

test_constraint_edible = SyntheticDivergentAssociationTest(
    models=[
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "ollama/gemma3n:e4b"
        ],
    configs=[
        {"temperature": 1},
    ],
    n_words=10,
    repeats=15,
    delay=11,
    constraints=language_constraints_edible,
    file_name="SDAT_1.3_onlyGoogle_eatable_constrained"
)

time.sleep(5)

test_constraint_tools = SyntheticDivergentAssociationTest(
    models=[
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "ollama/gemma3n:e4b"
        ],
    configs=[
        {"temperature": 1},
    ],
    n_words=10,
    repeats=15,
    delay=11,
    constraints=language_constraints_tools,
    file_name="SDAT_1.3_onlyGoogle_tools_constrained"
)

time.sleep(5)

test_constraint_money = SyntheticDivergentAssociationTest(
    models=[
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "ollama/gemma3n:e4b"
        ],
    configs=[
        {"temperature": 1},
    ],
    n_words=10,
    repeats=15,
    delay=11,
    constraints=language_constraints_money,
    file_name="SDAT_1.3_onlyGoogle_money_constrained"
)

time.sleep(5)

test_constraint_sound = SyntheticDivergentAssociationTest(
    models=[
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "ollama/gemma3n:e4b"
        ],
    configs=[
        {"temperature": 1},
    ],
    n_words=10,
    repeats=15,
    delay=11,
    constraints=language_constraints_sound,
    file_name="SDAT_1.3_onlyGoogle_sound_constrained"
)

print(test_constraint_edible)
print(test_constraint_tools)
print(test_constraint_money)
print(test_constraint_sound)

print("Starting requests!")
prev_edible = test_constraint_edible.request()
prev_tools = test_constraint_tools.request()
prev_money = test_constraint_money.request()
prev_sound = test_constraint_sound.request()

print("Cleaning LLM responses!")
prev_edible = test_constraint_edible.clean_llm_response(prev_edible)
prev_tools = test_constraint_tools.clean_llm_response(prev_tools)
prev_money = test_constraint_money.clean_llm_response(prev_money)
prev_sound = test_constraint_sound.clean_llm_response(prev_sound)

print("Calculating Embeddings!")
prev_edible = test_constraint_edible.calculate_embeddings(prev_edible)
prev_tools = test_constraint_tools.calculate_embeddings(prev_tools)
prev_money = test_constraint_money.calculate_embeddings(prev_money)
prev_sound = test_constraint_sound.calculate_embeddings(prev_sound)

print("Finished Test!")