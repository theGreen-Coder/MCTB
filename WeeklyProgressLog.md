# Weekly Progress Log
## Week #0 (Dates: 26th of May - 1st of June)
- **Goals for this Week:**
  - Finish Kaggle 5-Day Gen AI Intensive Course
  - Research already existing creative thinking benchmarks
  - Investigate Gemini 2.5 Flash
  - First Meeting with Mentor

- **Tasks Completed This Week:**
  - Set up Initial GitHub repo
  - Update Google Doc with Goals and Timeline
  - Research already existing creative thinking benchmarks
  - First Meeting with Mentor
  - Investigated Gemini 2.5 Flash

## Week #1 (Dates: 2nd of June - 8th of June)
- **Goals for this Week:**
  - Set up a basic evaluation script for existing creative thinking benchmarks
  - Familarisation with Gemini prompt engineering for creative benchmarks

- **Tasks Completed This Week:**
  - Set up a basic evaluation script for existing creative thinking benchmarks [(commit)](https://github.com/theGreen-Coder/MCTB/commit/4c166aa5a9f7d93050a54ef4e381574e73205e78)
  - Setup basic request script to make API calls to the Google API [(commit)](https://github.com/theGreen-Coder/MCTB/commit/546ef73a11f3f00f170d492bd4b9fc0ddc1f65f5)
  - Discovered LangChain and made some pretty major refactors to my code [(commit)](https://github.com/theGreen-Coder/MCTB/commit/aff0ec09da5e2786aa96b6397ca70a90a2cb03b4)

- **Challenges/Blockers Encountered:**
  - The main challenge was how to structure the object oriented design of the repository to be able to evaluate LLMs easily

- **Learnings/Discoveries:**
  - Learnt about LangChain, which gave me an easier way to integrate everything

## Week #2 (Dates: 9th of June - 15th of June)
- **Goals for this Week:**
  - Implement the divergent association task (DAT)
  - Preliminary evaluation with Gemini Models

- **Tasks Completed This Week:**
  - Initial first fully working DAT [(commit)](https://github.com/theGreen-Coder/MCTB/commit/64bb880db11fa48f2303defb76b4158c3b836a54)
  - Refactor of code + First evaluation of models [(commit)](https://github.com/theGreen-Coder/MCTB/commit/7e3620f1fae14603926c9fbb44ca81a24634c8fc), [(commit)](https://github.com/theGreen-Coder/MCTB/commit/d4904a55a6931d9a0d36c3aa11990ee461bcd4de)
  - Added multiple prompts and worked on preliminary code to plot results [(commit)](https://github.com/theGreen-Coder/MCTB/commit/48a388c3a2efbf1b8c5cde08b426fbe82f48d649)

- **Challenges/Blockers Encountered:**
  - Main challenge was the unreliability of LLM to output only words. An initial parsing function was implemented, but it was not enough.

- **Learnings/Discoveries:**
  - Gemma seems to always repeat the same words (but they scored pretty high)

## Week #3 (Dates: 16th of June - 22nd of June)
- **Goals for this Week:**
  - Better parsing function to clean out weird LLM responses
  - Adding BERT embedding model

- **Tasks Completed This Week:**
  - Refactor of plot.py and added correlation ploy [(commit)](https://github.com/theGreen-Coder/MCTB/commit/66454d3813d38e683dfbf11360018f5f9bfc71f0), [(commit)](https://github.com/theGreen-Coder/MCTB/commit/374ce140fd82e86827e5aac1fdbded4c1e8b8709)
  - Improved parsing function [(commit)](https://github.com/theGreen-Coder/MCTB/commit/382d416ea79c35bc88c76fb2b28f56dc4b220061#diff-bcd6d25acf728d8e9dbaf7cc0127190f0d6d9387753dcc8b06a455731fd37e95)
  - Minor modification to file names [(commit)](https://github.com/theGreen-Coder/MCTB/commit/5d56ac1926b05ba37fcc65269aa1e386ab603d29), [(commit)](https://github.com/theGreen-Coder/MCTB/commit/2c8f0f878ca5d0cc9f914870d8a1150237a5a4d6)
  - Tried adding BERT embedding model [(commit)](https://github.com/theGreen-Coder/MCTB/commit/ebb6e59104c23192d8b333030e82616df2c1c8b7), [(commit)](https://github.com/theGreen-Coder/MCTB/commit/d65239215dcd3a746443a751d8627661ceea874a)

- **Challenges/Blockers Encountered:**
  - Pretty major challenge of this week was adding BERT (still not solved)

- **Learnings/Discoveries:**
  - Learned about tokenisation, mean average pooling, and attention pooling through BERT and the transformers library

## Week #4 (Dates: 23rd of June - 29th of June)
- **Goals for this Week:**
  - Fix Speed of BERT embeddings
  - Implement Hard DAT
  - Implement DSI

- **Tasks Completed This Week:**
  - Finally managed to export all word embeddings of BERT to a h5 file [(commit)](https://github.com/theGreen-Coder/MCTB/commit/e04f5e8127a453320e17773ad401938d18693748)
  - Fixed a Major bug with the thinking_budget config

- **Tasks In Progress (and % complete or next steps):**
  - Working on a major refactor of the code + Implementation of HardDAT - 90%

- **Challenges/Blockers Encountered:**
  - The main challenge has been Google API rate limits (HardDAT has a lot of input tokens, which limits my API usage)
  - I didn’t update my pip3 packages, and I was working with an outdated version of langchain-google-genai

- **Learnings/Discoveries:**
  - BERT vs GloVe embeddings seem to affect the results slightly