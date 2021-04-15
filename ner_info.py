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


def query_df(df, speaker, speech):
    """Function to query a dataframe with the given pararmeters"""
    return df.query("speaker == @speaker and text == @speech")


def display_ner_data(df):
    st.markdown("# Natural Entity Recognition (NER)")

    st.markdown("""
    NER was used to extract the various entities involved in the discussion.
    We finetuned 2 models (`xlm-roberta-base` and `xlm-roberta-base-ontonotes5`) using manually annotated hansard data to extract the following entity types `PERSON, NORP, FAC, ORG, GPE, LAW, DATE`.

    Models were evaluated on Singapore Hansard NER Dataset validation set.

    | Model                                     | F1 Score | Precision | Recall | Remark     |
    |-------------------------------------------|----------|-----------|--------|------------
    | asahi417/tner-xlm-roberta-base-ontonotes5 | 0.343    | 0.274     | 0.458  |Pretrained model without finetuning|
    | xlm-roberta-base-sh-ner                   | 0.786    | 0.742     | 0.837  | Pretrained xlm-roberta-base model finetuned on the manually annotated Singapore hansard dataset|
    | xlm-roberta-base-ontonotes5-sh-ner        | **0.819**| **0.778** | **0.864** | Pretrained xlm-roberta-base-ontonotes5 model finetuned on the manually annotated Singapore hansard dataset|

    ### View our model results using the buttons below:
    """)
    col1, col2 = st.beta_columns([.35, 1])

    with col1:
        speakers = df['speaker'].unique()
        speaker_choice = st.selectbox('Select speaker:', speakers, index=47)
    with col2:
        speeches = df['text'].loc[df["speaker"] == speaker_choice].unique()
        speech_choice = st.selectbox('Select Speech', speeches, index=2)

    df_filtered = query_df(df, speaker_choice, speech_choice)

    doc = {"text": df_filtered['text'].values[0],
           "ents": ast.literal_eval(df_filtered['entities'].values[0])}
    HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
    html = displacy.render(doc, style="ent", manual=True)
    html = html.replace("\n\n", "\n")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    sequence = """Enter some Parliamentary Hansard text here to extract various entities like names (Heng Swee Kwat, K Shanmugam),
    places(Singapore, Malaysia), dates (12th May, 2012-02-03), organisations (SAF, MOH, MHA, Ministry of Finance),
    laws (Adoption of Children Act, Work Injury Compensation Act 2019)
    """

    st.markdown("## **Live Inference**")
    st.markdown("### Try out our NER Model `xlm-roberta-base-ontonotes5-sh-ner`")
    text_box = st.text_area("Enter some text for NER", sequence)

    HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

    if text_box:
        url = 'https://hansard-nlp-api-l6lhxur2aq-uc.a.run.app/ner/'
        req_body = json.dumps({'hansard_text': text_box, 'output': {}})
        response = requests.post(url, data=req_body)
        response = json.loads(response.text)['output']

        html = displacy.render(response, style="ent", manual=True)
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
