# Multimodal Creative Thinking Benchmark (MCTB)
An open-source multimodal creative thinking benchmark developed as part of Google Summer of Code (GSoC) under the mentorship of Google DeepMind ([@google-deepmind](https://github.com/google-deepmind)).

## Project Goals
The goal of this project is to develop a multi-modal and open-source benchmark with which to evaluate Gemini 2.0. Open-source benchmarks are a mostly unbiased way to test the ability of LLMs in different modalities. Current LLM benchmarks are mostly based on reasoning tasks and maths. 

However, there does not exist a gold standard LLM benchmark to assess creative thinking. This is a critical gap since creative thinking is essential for progress toward artificial general intelligence (AGI). To solve this, the project aims to extend current creative thinking LLM benchmarks and create a gold standard benchmark across four different modalities (text, image, video, and audio). 

Upon the project’s completion, a thorough multi-modal open-source benchmark for creative thinking and creative divergence will be created. This will be accompanied by the evaluation metrics of different models (including Gemini 2.0), along with documentation and custom evaluation scripts. Finally, an educational video explanation of the benchmark will be shared on my YouTube channel (https://www.youtube.com/@Green-Code/), which will serve as an introduction to developers and newcomers to the LLM field.

## Quick-start
### Start by cloning the repo:

```
git clone https://github.com/theGreen-Coder/MCTB.git
cd MCTB
```
### Create & activate the Conda environment:

```
conda env create -f environment.yml   # installs Python 3.12 and all pinned deps  
conda activate MCTB                   # environment name is set inside the YAML
```

Conda’s env create -f command builds an environment from the yml file.

### Add your API keys
```
cp .env-example .env   # or copy it manually on Windows
```

Open .env and replace the placeholders:
```
OPENAI_API_KEY=YOUR_KEY_HERE
GOOGLE_API_KEY=YOUR_KEY_HERE
```
These are the only two variables required according to .env-example. (github.com)

### Get started

Check out [the example notebook](example.ipynb) to see my main results so far in the project, as well as a general explanation of each of the test and it's implementation.
Otherwise, all creative tests are stored in the [creative_tests folder](creative_tests/).\
Hope you enjoy!

### Additional Notes
A couple of really big files are needed to fully execute the repository. You will need to download `glove.840B.300d.txt`:

```
wget -P models http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip models/glove.840B.300d.zip -d models
```

Also, the file `bert_midlayer_dict.h5` is missing. This is roughly a 1GB file that I genereted to get the word embeddings faster using BERT (by storing them beforehand). At the moment (June 26th 2025), this is still a bit buggy, so I will upload this when I fix all the problems.

## Weekly Progress Log
A weekly updated progress log can be found [here](WeeklyProgressLog.md).

## Acknowledgements
Thank you to Paige Bailey for her mentorship.

## Community contributions
Contributions are welcome—feel free to open issues or submit pull requests!

## License
This project is licensed under the MIT License. A copy of the MIT License can be found [here](LICENSE).

GloVe model in embeddings.py is based on code from [this repo](https://github.com/jayolson/divergent-association-task), originally by Jay Olson. Modified under [LICENSE](https://github.com/jayolson/divergent-association-task/blob/main/LICENSE.txt).