# Singapore Parliament Hansard NLP Demo

Analysis of Singpore's parliamentary hansard using NLP (NER, Sentiment Analysis, Summarisation).  
Deployed with Streamlit - [Demo](https://share.streamlit.io/nus-cs3244-ml-singapore-7/ner-demo/app.py)

Compiled Demo of Parliament NLP tasks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nus-cs3244-ml-singapore-7/NER-demo/blob/master/Parliament_Hansard_NLP_CS3244.ipynb) 


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
  
