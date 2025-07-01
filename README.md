# Unlearn Stress Testing LangGraph

## Introduction

Based on [google-gemini/gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)

Main modifications made:
1. Support for local LLMs (e.g., starting a Azure OpenAI API with lm-studio), while changing Gemini's API interface to OpenAI-compatible format [This code was entirely modified with Codex's help]
2. Tool calling changed to use Google Search API for searching (relatively slow, DuckDuckGo is also available, there's DDGS-related code in the codebase, but it often hits rate limits...) [This code was modified based on Codex's changes, referencing the Google Search API from [https://github.com/leptonai/search_with_lepton/blob/main/search_with_lepton.py](https://github.com/leptonai/search_with_lepton/blob/main/search_with_lepton.py). I just copied this since I had tried Google's before. Of course, there are other search APIs available that can be tried later]
3. For convenient local debugging, removed some Docker-related components

## Usage

1. Python version must be 3.11 or above
2. Obtain two parameters for Google Search API
    + api-key: [link](https://developers.google.com/custom-search/v1/introduction?hl=zh-cn#identify_your_application_to_google_with_api_key)
    + cx: [link](https://stackoverflow.com/questions/6562125/getting-a-cx-id-for-custom-search-google-api-python)
3. LANGSMITH_API_KEY is optional, can be obtained from the LangChain official website
4. Reference ```backend/.env.example``` to create ```backend/.env```, uncomment all lines and modify to your base_url (for lm-studio, it's the default port 1234), openai_api_key (for local services, just write anything), google api-key, cx and LANGSMITH_API_KEY
5. Go to the backend directory and run ```pip3 install .```
6. Install node.js, go to the frontend directory and run ```npm install```
7. Return to the main directory, run ```make dev```, and open ```http://localhost:5173/app/``` to use

## Features

### Online Search with Google

### RAG Integration




## Star History

<a href="https://star-history.com/#YuyangXueEd/unlearn_stress_testing_langgraph&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=YuyangXueEd/unlearn_stress_testing_langgraph&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=YuyangXueEd/unlearn_stress_testing_langgraph&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=YuyangXueEd/unlearn_stress_testing_langgraph&type=Date" />
 </picture>
</a>

