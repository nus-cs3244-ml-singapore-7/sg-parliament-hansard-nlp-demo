# Singapore Parliament Hansard NLP Demo

Demo website for for [CS3244 Machine Learning](learning) Project (AY20/21 Semester 2).

## About
The project was to perform analysis of Singpore's parliamentary hansard using NLP. We trained and ran Transformer models on the Singapore Parliament
Hansard to perform sentiment analysis, name-entity recognition and summarisation, taking advantage of transfer
learning via pre-trained models.

The models were used to analyse the Hansard of recent years to uncover interesting findings on speakers and entities by sentiment. The findings can be found on the [demo website](https://share.streamlit.io/nus-cs3244-ml-singapore-7/ner-demo/app.py).

## Project Report
The project report can be found [here](https://github.com/nus-cs3244-ml-singapore-7/sg-parliament-hansard-nlp-demo/blob/master/transfer-learning-with-transformers-machine-analysis-of-the-singapore.pdf).

## Data
The data for training and analysed was scraped from the Singapore Parliament website for all sessions from September 2012 to March 2021, giving about a decade worth of information and spanning three sessions of Parliament. The data can be found [here](https://github.com/nus-cs3244-ml-singapore-7/hansard_data).

## Model Training
The trained models along with the code and results for model training can be found [here](https://github.com/nus-cs3244-ml-singapore-7/singapore-hansard-nlp).

This notebook can be used to get a quick overview of the NLP tasks using [Hugging Face Transformers](https://huggingface.co/transformers/) library:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nus-cs3244-ml-singapore-7/NER-demo/blob/master/Parliament_Hansard_NLP_CS3244.ipynb) 

## Deployment
- The models for sentiment analysis and name-entity recognition were deployed on Google cloud run using [FastAPI](https://fastapi.tiangolo.com/) and [Docker](https://www.docker.com/). Deployment code and details can be found [here](https://github.com/nus-cs3244-ml-singapore-7/sg-hansard-nlp-api).
- The demo website was deployed using [Streamlit](https://streamlit.io/). Instructions for setting up the website locally are given below.


## Running Locally
1. Clone the repo and navigate to the correct folder

  ```
  git clone https://github.com/nus-cs3244-ml-singapore-7/NER-demo.git
  cd NER-demo
  ```

2. Create a virtual environment

  ```
  pip install virtualenv #Run this if you don't have virtualenv installed
  virtualenv env
  ```

3. Activate the virtual environment

  ```
  env\Scripts\activate #Windows
  source env/lib/activate #Mac/Linux
  ```
  
4. Install the project requirements

  ```
  pip install -r requirements.txt
 ```  
5. Run `streamlit run app.py`
6. Visit `localhost:8501` to view the app
  
## Built With
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)

## Team
- [Joel Ho Eng Kiat](https://github.com/JoelHo)
- [Kingsley Kuan Jun Hao](https://github.com/kingsleykuan)
- [Neo Neng Kai Nigel](https://github.com/nigelnnk)
- [Niveditha Nerella](https://github.com/nivii26)
- [Noel Mathew Isaac](https://github.com/noelmathewisaac)
- [Timothy Ong Jing Kai](https://github.com/timjkong)
