import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from request import Request, run_request
from langchain_ollama import ChatOllama
from sklearn.utils import shuffle
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

### HYPERparams ###
OBJECT_NAME = "brick"
N_PROMPTS = 300
N_EXAMPLES = 10

def build_prompt_for_object(object_name, uses, ratings_prefilled, target_index, to_evaluate_use):
    # 1-based numbering so ratings and target_index align
    uses_lines = [f"{i}. {u}" for i, u in enumerate(uses, start=1)]
    
    ratings_lines = []
    for i in range(1, len(uses) + 1):
        if i in ratings_prefilled:
            ratings_lines.append(f"{i}. {ratings_prefilled[i]}")
        else:
            ratings_lines.append(f"{i}.")
    
    return template_flexible.format(
        object=object_name,
        uses_block="\n".join(uses_lines),
        ratings_block="\n".join(ratings_lines),
        rating_target_index=target_index,
        to_evaluate_use=to_evaluate_use
    )

df = pd.read_csv("data/organisciak_2023/all_data.csv")

template_flexible = """
Below is a list of uses for {object}. On a scale of 10-50, judge how original each use for {object} is, 
where 10 is "not at all creative" and 50 is "very creative":

USES
{uses_block}
{rating_target_index}. {to_evaluate_use}

RATINGS
{ratings_block}
{rating_target_index}.

Output only the rating for use #{rating_target_index}. Only the number.
""".strip()

object_name_responses = df[df["prompt"] == OBJECT_NAME]
object_name_responses = shuffle(object_name_responses, random_state=41)

prompt_uses_first_10 = object_name_responses.head(N_EXAMPLES)
object_name_responses_test = object_name_responses.iloc[N_EXAMPLES:N_PROMPTS+N_EXAMPLES]

final_prompt_object_name = build_prompt_for_object(
    OBJECT_NAME.upper(),
    prompt_uses_first_10["response"].to_list(),
    {i+1: int(x * N_EXAMPLES) for i, x in enumerate(prompt_uses_first_10["target"].to_list())},
    target_index=11,
    to_evaluate_use="{use_input}"
)

print(final_prompt_object_name)

model = ChatOllama(model="gemma3n:e4b")  # must have been `ollama pull`-ed locally
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are an expert alternative uses test (AUT) rater."), ("human", final_prompt_object_name)]
)

inputs = []
for use in object_name_responses_test["response"].to_list():
    pv = prompt.invoke({"use_input": use})
    inputs.append(pv.to_string())

print("Printing 1st Prompt")
print(inputs[0])
print()

print("Printing last prompt")
print(inputs[N_PROMPTS-1])
print()
print(f"Len inputs: {len(inputs)}")

non_clean_llm_response = []
targets = object_name_responses_test["target"].to_list()
uses_to_evaluate = object_name_responses_test["response"].to_list()

# Request API
for idx, prompt_input in tqdm(enumerate(inputs)):
    SDAT_request = Request(
        models=[
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "ollama/gemma3n:e4b",
        ],
        prompt=prompt_input,
        configs={"temperature": 0.5},
        repeats=3,
        default_delay=1,
        verbose=False
    )

    llm_response = run_request(SDAT_request)
    llm_response["config"] = {}
    llm_response["config"]["target"] = targets[idx]
    llm_response["config"]["prompt"] = uses_to_evaluate[idx]

    non_clean_llm_response.append(llm_response)

    print("Saving to file!")
    with open(f"data/correlation_models_{OBJECT_NAME}_{str(N_PROMPTS)}prompts.json", "w") as json_file:
        json.dump(non_clean_llm_response, json_file, indent=4)