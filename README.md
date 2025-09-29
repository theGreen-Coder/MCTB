# Multimodal Creative Thinking Benchmark (MCTB)
An open-source multimodal creative thinking benchmark developed as part of Google Summer of Code (GSoC) under the mentorship of Google DeepMind ([@google-deepmind](https://github.com/google-deepmind)). See [the GSoC notebook](GSoC25.md) for the evaluation and results obtained during GSoC 2025.

## Table of Contents

- [Project Goals](#project-goals)  
- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
  - [Download Required Files](#download-required-files)  
- [Usage & Examples](#usage--examples)  
- [Contributing](#contributing)  
- [Acknowledgements](#acknowledgements)  
- [License](#license)

## Project Goals
- Build a multi-modal open-source benchmark to evaluate creative thinking (not just reasoning) in large language and multimodal models. Specifically, MCTB is focued on the evaluation of divergent thinking capabilites of LLMs.
- Adapt common psychometric test of divergent thinking. For example:
    - Divergent Association Task (DAT)
    - Hard Divergent Association Task (HardDAT)
    - Synthetic-Divergent Association Task (S-DAT)
    - Alternative Uses Test (AUT)
- Provide **evaluation scripts, metrics, and baseline results**. 

## Features

- Several divergent thinking tests already implemented (see `creative_tests/`)  
- Support for local and API-based inference  
- Evaluation scripts and reporting utilities  
- Support for several word and sentence embedding models (e.g. GloVe, BERT, SBERT, etc., see see `embeddings.py`)  
- Open-source and initial result evaluations (see `GSoC25.md`)

## Getting Started
Check out [the GSoC notebook](GSoC25.md) to see my main results so far in the project, as well as a general explanation of each of the test and it's implementation.
Otherwise, all creative tests are stored in the [creative_tests folder](creative_tests/).

### Prerequisites

- **Python** — version specified in `environment.yml`  
- **Conda** — for managing environments easily  
- Access to desired LLM APIs: Most support is for Gemini models, but there is also minimal support for OpenAI and Anthropic models. In addition, `MCTB` supports local inference using Ollama

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/theGreen-Coder/MCTB.git
cd MCTB

# 2. Create & activate Conda environment
conda env create -f environment.yml   # installs Python 3.12 and all pinned deps  
conda activate MCTB                   # environment name is set inside the YAML
```
Conda’s env create -f command builds an environment called MCTB from the yml file.

### Configuration
```
cp .env-example .env   # or copy it manually on Windows
```

Open .env and replace the placeholders:
```
OPENAI_API_KEY=YOUR_KEY_HERE
GOOGLE_API_KEY=YOUR_KEY_HERE
ANTHROPIC_API_KEY=YOUR_KEY_HERE
```
These are the only two variables required according to .env-example. (github.com)

### Download Required Files (to run all notebooks)
To access the main functionalities of the repo, a couple of big files are needed (all stored in `models/`): 

1. You will need to download `glove.840B.300d.txt`:
    ```
    wget -P models http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip models/glove.840B.300d.zip -d models
    ```
2. BERT embedding model
3. SBERT model
4. Granite-embedding-278m-multilingual
5. Other less used embedding models if desired (`FastText`, `Word2Vec`, etc.)

#### Note on local inference
Local inference can be used using Ollama. When running just input `ollama/your-desired-model` into the `models` argument of the creative tests and [request.py](request.py) should take care of everything.

### Usage & Examples
The API to run and test the implemented `creative_tests` is the following:

#### DAT Example

```python
from creative_tests import DivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

DAT_test = DivergentAssociationTest(
    models=[
            "ollama/gemma-3n-e4b-it", # Support for local inference using ollama
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
        ],
    configs=[ # Different temperature configs (or top_p, thinking_budget)
        {"temperature": 0.5},
        {"temperature": 1},
        {"temperature": 1.5},
        {"temperature": 2},
    ],
    n_words=20, # Number of words for the LLM to generate during DAT
    repeats=50, # Number of experimental repeats
    delay=11, # Delay between API calls if no specific delay in specified in models/time_delay_models_config.json
    standard_prompt=False, # If False it enables prompt variations
    file_name="DAT_1.2_robust_DAT_diffTemp_diffPrompts" # Name of the file when results are stored
)

DAT_test.run() # Runs the DAT pipeline specified in test
```

#### HardDAT Example
```python
from creative_tests import HardDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

HardDAT_test = HardDivergentAssociationTest(
    models=[
            "ollama/gemma-3n-e4b-it", # Support for local inference using ollama
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
        ],
    configs=[ # Different temperature configs (or top_p, thinking_budget)
        {"temperature": 0.5},
        {"temperature": 1},
        {"temperature": 1.5},
        {"temperature": 2},
    ],
    n_words=25, # Number of words for the LLM to generate during DAT
    given_words=20, # Number of words given in hardDAT as input
    common=True, # Sample given words from models/5k_common.txt
    delay=11, # Delay between API calls if no specific delay in specified in models/time_delay_models_config.json
    file_name="DAT_1.2_robust_HardDAT_diffTemp_diffPrompts" # Name of the file when results are stored
)

HardDAT_test.run() # Runs the DAT pipeline specified in test
```

#### S-DAT Example
```python
from creative_tests import SyntheticDivergentAssociationTest
from dotenv import load_dotenv

load_dotenv()

# Example for testing the effect of thinking_budget using gemini-2.5-flash-lite
SDAT_test = SyntheticDivergentAssociationTest(
    models=["gemini/gemini-2.5-flash-lite"],
    configs = [
                {"temperature": 1, "thinking_budget": 512}, # Specify desired thinking_budget configs
                {"temperature": 1, "thinking_budget": 768},
                {"temperature": 1, "thinking_budget": 1024},
                {"temperature": 1, "thinking_budget": 1536},
                {"temperature": 1, "thinking_budget": 2048},
                {"temperature": 1, "thinking_budget": 3072},
                {"temperature": 1, "thinking_budget": 4096},
                {"temperature": 1, "thinking_budget": 6144},
                {"temperature": 1, "thinking_budget": 8192},
                {"temperature": 1, "thinking_budget": 12288},
                {"temperature": 1, "thinking_budget": 24000},
    ],
    n_words=10, # Number of words for the LLM to generate during S-DAT
    repeats=15, # Number of experimental repeats
    delay=11, # Delay between API calls if no specific delay in specified in models/time_delay_models_config.json
    file_name="SDAT_1.2_thinking_budget_flash" # Name of the file when results are stored
)

SDAT_test.run()
```

#### DSI Example
```python
from creative_tests import DivergentSemanticIntegration
from dotenv import load_dotenv

load_dotenv()

# Simple example for DSI test evaluating only Gemini models
DSI_test = DivergentSemanticIntegration(
    models=["gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "gemma/gemma-3n-e4b-it"
        ],
    configs=[
        {"temperature": 1},
    ],
    repeats=3, # 3 * 8 (number of cue words) = 24 repeats
    delay=15, # Delay between API calls if no specific delay in specified in models/time_delay_models_config.json
)

DSI_test.run()
```

#### AUT Example (still a bit unstable)
```python
from creative_tests import SimpleAlternativesUseTask
from dotenv import load_dotenv

load_dotenv()

# Extensive AUT test using most supported models
AUT_test = SimpleAlternativesUseTask(
    models=[
            "ollama/gemma3n:e4b",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro", 
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "claude/claude-sonnet-4-20250514",
            "gpt/gpt-5"
        ],
    configs=[
        {"temperature": 0.7}, # All models evaluated at 0.7 temperature
    ],
    target_objects = ["brick", "box", "paperclip", "bottle"], # Common object to be prompted during AUT
    n_uses = 20, # Number of uses to be answered by LLM
    repeats = 25, # Number of experimental repeats
    standard_prompt = False, # Number of experimental repeats
    disallow_common_uses = True, # Prompt LLMs to avoid common/expected uses
    delay = 1, #  Delay between API calls if no specific delay in specified in models/time_delay_models_config.json
    file_name="AUT_1.0_all_models", # File name
)

AUT_test.run()
```

**Note**\
If any of these test are unclear, please check the folder `testing/` for more examples.

## Contributing
Contributions are welcome. Feel free to open issues or submit pull requests!

## Acknowledgements
A special thank you to:  
- [Paige Bailey](https://github.com/dynamicwebpaige) for her insights and huge help through the whole duration of this project.
- [Xavier Amatriain](https://www.linkedin.com/in/xamat/) for his valuable insights and generosity with his time.
- My fellow GSoC DeepMind contributors for their feedback and inspiration (especially [@rorosaga](https://github.com/rorosaga)).
- Anyone reading this for taking the time to check out my project! :)

## License
This project is licensed under the MIT License. A copy of the MIT License can be found [here](LICENSE).

GloVe model in embeddings.py is based on code from [this repo](https://github.com/jayolson/divergent-association-task), originally by Jay Olson. Modified under [LICENSE](https://github.com/jayolson/divergent-association-task/blob/main/LICENSE.txt).