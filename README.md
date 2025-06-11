# Multimodal Creative Thinking Benchmark (MCTB)
An open-source multimodal creative thinking benchmark developed as part of Google Summer of Code (GSoC) under the mentorship of Google DeepMind ([@google-deepmind](https://github.com/google-deepmind)).

## Project Goals
The goal of this project is to develop a multi-modal and open-source benchmark with which to evaluate Gemini 2.0. Open-source benchmarks are a mostly unbiased way to test the ability of LLMs in different modalities. Current LLM benchmarks are mostly based on reasoning tasks and maths. 

However, there does not exist a gold standard LLM benchmark to assess creative thinking. This is a critical gap since creative thinking is essential for progress toward artificial general intelligence (AGI). To solve this, the project aims to extend current creative thinking LLM benchmarks and create a gold standard benchmark across four different modalities (text, image, video, and audio). 

Upon the project’s completion, a thorough multi-modal open-source benchmark for creative thinking and creative divergence will be created. This will be accompanied by the evaluation metrics of different models (including Gemini 2.0), along with documentation and custom evaluation scripts. Finally, an educational video explanation of the benchmark will be shared on my YouTube channel (https://www.youtube.com/@Green-Code/), which will serve as an introduction to developers and newcomers to the LLM field.

## Weekly Progress Log
A weekly updated progress log can be found [here](WeeklyProgressLog.md).

## Acknowledgements
Thank you to Paige Bailey for her mentorship.

## Community contributions
Contributions are welcome—feel free to open issues or submit pull requests!

## License
This project is licensed under the MIT License. A copy of the MIT License can be found [here](LICENSE).

GloVe model in embeddings.py is based on code from [this repo](https://github.com/jayolson/divergent-association-task), originally by Jay Olson. Modified under [LICENSE](https://github.com/jayolson/divergent-association-task/blob/main/LICENSE.txt).