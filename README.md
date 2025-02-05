---
title: Open Deep-Research
emoji: 🏆
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: OpenAI's Deep Research, but open
---

> NOTE: Open Deep Research provided from [SmolAgents](https://github.com/huggingface/smolagents)!

# Running Locally:

1. Clone the repository:

```bash
git clone https://github.com/AI-Maker-Space/DeepResearch-HF.git
cd open-deep-research
```

2. Install dependencies:

```bash
uv sync
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY="your_openai_key"
export SERPAPI_API_KEY="your_serpapi_key"
export HF_TOKEN="your_huggingface_token"
```

4. Run the application:

```bash
uv run app.py
```




