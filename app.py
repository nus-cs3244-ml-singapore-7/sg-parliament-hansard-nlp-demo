from spacy import displacy
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import ast
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px
from sentiment_info import *
from ner_info import *


@st.cache()
def get_data():
    df = pd.read_csv("./Combined Data 14th.csv")
    return df


# Load Data
st.title("Singapore Parliament Hansard NLP")
st.write("""
The Singapore Parliament Hansard contains verbatim transcripts of the speeches made by politicians in official parliament sessions.
We analysed the Hansard using three different NLP tasks to gain more insights into the parlimentary proceedings.
The data anaylsed was scraped from the Singapore Parliament website for all sessions from September 2012 to March 2021, giving about a decade worth of information and spanning three sessions of Parliament.

Below is the data for the 14th session of parliament along with some interesting insights and visualisations drawn from the data.
""")
df = get_data()
cols = df.columns.tolist()

st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)
st.write("Each row pertains to a speech given by a person")
st.dataframe(df[st_ms])

display_ner_data(df)
display_sentiment_data(df)
