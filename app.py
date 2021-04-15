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


@st.cache
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

# st.markdown("# Natural Entity Recognition (NER)")

# st.markdown("""
# NER was used to extract the various entities involved in the discussion.
# We finetuned 2 models (`xlm-roberta-base` and `xlm-roberta-base-ontonotes5`) using manually annotated hansard data to extract the following entity types `PERSON, NORP, FAC, ORG, GPE, LAW, DATE`.

# Models were evaluated on Singapore Hansard NER Dataset validation set.

# | Model                                     | F1 Score | Precision | Recall | Remark     |
# |-------------------------------------------|----------|-----------|--------|------------
# | asahi417/tner-xlm-roberta-base-ontonotes5 | 0.343    | 0.274     | 0.458  |Pretrained model without finetuning|
# | xlm-roberta-base-sh-ner                   | 0.786    | 0.742     | 0.837  | Pretrained xlm-roberta-base model finetuned on the manually annotated Singapore hansard dataset|
# | xlm-roberta-base-ontonotes5-sh-ner        | **0.819**| **0.778** | **0.864** | Pretrained xlm-roberta-base-ontonotes5 model finetuned on the manually annotated Singapore hansard dataset|

# ### View our model results using the buttons below:
# """)

# col1, col2 = st.beta_columns([.35, 1])

# with col1:
#     speakers = df['speaker'].unique()
#     speaker_choice = st.selectbox('Select speaker:', speakers, index=47)
# with col2:
#     speeches = df['text'].loc[df["speaker"] == speaker_choice].unique()
#     speech_choice = st.selectbox('Select Speech', speeches, index=2)

# df_filtered = query_df(speaker_choice, speech_choice)

# doc = {"text": df_filtered['text'].values[0],
#        "ents": ast.literal_eval(df_filtered['entities'].values[0])}
# HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
# html = displacy.render(doc, style="ent", manual=True)
# html = html.replace("\n\n", "\n")
# st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


# sequence = """ 
# Enter some Parliamnetary Hansard text here to extract various entities like names (Heng Swee Kwat),
# places(Singapore, Malaysia), dates (12th May, 2012-02-03), organisations (SAF, MOH, MHA, Minitry of Finance),
# laws (Adoption of Children Act, Work Injury Compensation Act 2019)
# """

# st.markdown("### Try out our NER Model `xlm-roberta-base-ontonotes5-sh-ner`")
# text_box = st.text_area("Enter some text for NER", sequence)

# HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# if text_box:
#     url = 'https://hansard-nlp-api-l6lhxur2aq-uc.a.run.app/ner/'
#     req_body = json.dumps({'hansard_text': text_box, 'output': {}})
#     response = requests.post(url, data=req_body)
#     response = json.loads(response.text)['output']

#     html = displacy.render(response, style="ent", manual=True)
#     html = html.replace("\n\n", "\n")
#     st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


# st.markdown("# Sentiment Analysis")

# st.markdown("""
# Sentiment analysis was used to the positive/negative sentiments of portions of speeches, which can be helpful to determine if the speaker is for or against the topic at hand
# We finetuned 2 models (`xlm-roberta-base` and `xlm-roberta-base-sst-2`) on the manually labelled Singapore Hansard sentiment dataset and the HanDeSet dataset.

# Models were evaluated on Singapore Hansard Sentiment Dataset validation set.

# | Model                                        | Accuracy | F1 Score | Precision | Recall |
# |----------------------------------------------|----------|----------|-----------|--------|
# | xlm-roberta-base-sst-2                       | 0.780    | 0.834    | 0.820     | 0.849  |
# | xlm-roberta-base-handeset                    | 0.447    | 0.547    | 0.587     | 0.512  |
# | xlm-roberta-base-sst-2-handeset              | 0.561    | 0.691    | 0.637     | 0.756  |
# | xlm-roberta-base-sh-sentiment                | 0.856    | 0.889    | 0.894     | **0.884**  |
# | xlm-roberta-base-sst-2-sh-sentiment          | **0.879**| **0.904**| **0.938** | 0.872  |
# | xlm-roberta-base-handeset-sh-sentiment       | 0.773    | 0.828    | 0.818     | 0.837  |
# | xlm-roberta-base-sst-2-handeset-sh-sentiment | 0.841    | 0.873    | 0.911     | 0.837  |

# """)

# cm = sn.light_palette("green", as_cmap=True)

# # Speaker Data
# speaker_pivot = make_pivot_table(df, ['speaker'], value_columns=['sentiment'],
#                                  agg_function=(np.average, sentence_count))
# st.markdown("## Sentiments by Speaker")
# st.write("Data aggregated to show average sentiment and total sentences spoken for each session")

# # Interactive widgets----------------------
# speaker_values = st.slider(
#     'Select a sentiment range',
#     0.0, 1.0, (0.0, 1.0), key="speaker")
# speaker_min = speaker_values[0]
# speaker_max = speaker_values[1]
# speaker_pivot = speaker_pivot.query(
#     "@speaker_min<=`sentiment average`<=@speaker_max")
# st.write("Average sentiment range: ", speaker_values)
# # ----------------------

# st.dataframe(speaker_pivot.style.background_gradient(cmap=cm))

# fig, ax = plt.subplots()
# fig = px.scatter(speaker_pivot, x="sentiment average", y="sentiment sentence_count",
#                  color="sentiment average", size="sentiment sentence_count", hover_data=['speaker'],
#                  color_continuous_scale=px.colors.sequential.Viridis)
# st.plotly_chart(fig)


# # Session Data
# session_pivot = make_pivot_table(df, ['session_title'], value_columns=['sentiment'],
#                                  agg_function=(np.average, sentence_count))
# st.markdown("## *Sentiments by Session*")
# st.write("Data aggregated to show average sentiment and total sentences spoken for each session")

# # Interactive widgets----------------------
# session_values = st.slider(
#     'Select a sentiment range',
#     0.0, 1.0, (0.0, 1.0), key="session")
# session_min = session_values[0]
# session_max = session_values[1]
# session_pivot = session_pivot.query(
#     "@session_min<=`sentiment average`<=@session_max")
# st.write("Average sentiment range: ", session_values)
# # ----------------------

# st.dataframe(session_pivot.style.background_gradient(cmap=cm))

# fig, ax = plt.subplots()
# fig = px.scatter(session_pivot, x="sentiment average", y="sentiment sentence_count",
#                  color="sentiment average", hover_data=['session_title'],
#                  color_continuous_scale=px.colors.sequential.Viridis)
# st.plotly_chart(fig)

# st.markdown("### Try out our Sentiment Analysis Model `xlm-roberta-base-sst-2-sh-sentiment`")

# sentiment_text = """ 
# ‘Green shoots’ of recovery for Singapore’s economy, although uncertainties remain: Economists
# """

# text_box = st.text_area("Enter some text for Sentiment Analysis", sentiment_text)

# if text_box:
#     url = 'https://hansard-nlp-api-l6lhxur2aq-uc.a.run.app/sentiment/'
#     req_body = json.dumps({'hansard_text': text_box, 'output': {}})
#     response = requests.post(url, data=req_body)
#     response = json.loads(response.text)['output']

#     st.write(response)
