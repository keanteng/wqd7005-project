# WQD7007 Project

![Static Badge](https://img.shields.io/badge/python-3.12-blue)
[![made-with-latex](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keanteng/wqd7005-project)

## Title

Harnessing AI and Language Models for Predictive Modeling of Patient Health Deterioration

## Objective

Leverage AI technologies, including Large Language Models (LLMs), Small Language Models (SLMs), and Generative AI (GenAI), to engineer features and build predictive models for anticipating patient health deterioration based on collected vital signs and health questionnaire responses.

## Powered By
[![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=fff)](#)
[![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-000?logo=githubcopilot&logoColor=fff)](#)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-886FBF?logo=googlegemini&logoColor=fff)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)

## What to Do?

1. Dataset Simulation and Feature Engineering (4 marks)
    - Utilize GenAI to generate realistic patient data, including vital signs and detailed textual questionnaire responses. 
    - Apply LLMs to extract meaningful features from simulated textual data (clinical notes, health records). 
2. Predictive Model Development (5 marks) 
    - Construct and evaluate predictive models, including traditional models (Random Forest, XGBoost, Neural Networks) and advanced Transformer-based models. 
    - Use SLMs for specialized NLP tasks like sentiment analysis, clinical text interpretation, and classification of questionnaire responses. 
3. Model Evaluation and Interpretation (4 marks) 
    - Evaluate model performance using metrics such as accuracy, F1-score, ROC-AUC, etc. 
    - Leverage LLMs to summarize and interpret complex model outputs, explaining performance clearly and concisely. 
4. Comprehensive AI-Assisted Final Report (2 marks) 
    - Provide a detailed report (5-7 pages) clearly outlining the methodology, analytical techniques, results comparison, conclusions, and future recommendations. 
    - Explicitly disclose the use of AI tools within your report. 

##  Using this Repository

Clone this repository:

```bash
git clone https://github.com/keanteng/wqd7005-project
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Model Hub

| Model Card | Description | Link |
:-- |:-- |:-- |
| Sentiment Analysis | A model trained to analyze sentiment in text data, particularly useful for understanding patient feedback. | [Sentiment Analysis Model](https://huggingface.co/keanteng/bert-sentiment-wqd7007) |
| Fatigue Text Classification | A model designed for classifying text data into predefined categories, aiding in the organization of patient responses. | [Text Classification Model](https://huggingface.co/keanteng/bert-fatigue-response-classification-wqd7005) |
| Mental Health Text Classification | A model that classifies text responses related to mental health, helping in the identification of patient concerns. | [Mental Health Text Classification Model](https://huggingface.co/keanteng/bert-mental-health-response-classification-wqd7005) |
| Named Entity Recognition | A model that identifies and classifies key entities in text, such as medical terms, enhancing data extraction from patient records. | [Named Entity Recognition Model](https://huggingface.co/keanteng/bert-ner-wqd7005) |