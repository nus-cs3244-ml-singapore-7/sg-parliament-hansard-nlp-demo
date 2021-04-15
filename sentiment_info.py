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


@st.cache
def make_pivot_table(df, index_columns, value_columns, agg_function=np.sum):
    """ creates a pivot table for the given dataframe
    input: df = dataframe to make the pivot table out of
           index_columns = single column or set of two columns to be used as
                           columns to group by
           value_columns = the columns to aggregate as a target in the pivot table
           agg_function = what to use for the aggregation; default=np.sum
    output: pivot_color = pivot table with color scheme based on values
    """
    # create pivot table with internal pandas function
    cm = sn.light_palette("green", as_cmap=True)
    pivot = pd.pivot_table(
        data=df,
        index=index_columns,
        values=value_columns,
        aggfunc=agg_function
    )

    # reset the index
    pivot = pivot.reset_index()

    # convert the value columns to float
    for i in value_columns:
        pivot[i] = pivot[i].astype('float')

   # sort by the values in the index columns
    pivot = pivot.sort_values(by=index_columns)
    pivot.columns = pivot.columns.map(' '.join).str.strip()

    # color the pivot table based on the target value
    pivot_color = pivot.style.background_gradient(cmap=cm)
    return pivot


def sentence_count(x):
    return int(len(x))


def display_sentiment_data(df):
    st.markdown("# Sentiment Analysis")

    st.markdown("""
    Sentiment analysis was used to the positive/negative sentiments of portions of speeches, which can be helpful to determine if the speaker is for or against the topic at hand.
    We finetuned 2 models (`xlm-roberta-base` and `xlm-roberta-base-sst-2`) on the manually labelled Singapore Hansard sentiment dataset and the HanDeSet dataset.

    Models were evaluated on Singapore Hansard Sentiment Dataset validation set.

    | Model                                        | Accuracy | F1 Score | Precision | Recall |
    |----------------------------------------------|----------|----------|-----------|--------|
    | xlm-roberta-base-sst-2                       | 0.780    | 0.834    | 0.820     | 0.849  |
    | xlm-roberta-base-handeset                    | 0.447    | 0.547    | 0.587     | 0.512  |
    | xlm-roberta-base-sst-2-handeset              | 0.561    | 0.691    | 0.637     | 0.756  |
    | xlm-roberta-base-sh-sentiment                | 0.856    | 0.889    | 0.894     | **0.884**  |
    | xlm-roberta-base-sst-2-sh-sentiment          | **0.879**| **0.904**| **0.938** | 0.872  |
    | xlm-roberta-base-handeset-sh-sentiment       | 0.773    | 0.828    | 0.818     | 0.837  |
    | xlm-roberta-base-sst-2-handeset-sh-sentiment | 0.841    | 0.873    | 0.911     | 0.837  |

    """)

    cm = sn.light_palette("green", as_cmap=True)

    # Speaker Data
    speaker_pivot = make_pivot_table(df, ['speaker'], value_columns=['sentiment'],
                                     agg_function=(np.average, sentence_count))
    st.markdown("## **Sentiments by Speaker**")
    st.write("Data aggregated to show average sentiment and total sentences spoken for each session")

    # Interactive widgets----------------------
    speaker_values = st.slider(
        'Select a sentiment range',
        0.0, 1.0, (0.0, 1.0), key="speaker")
    speaker_min = speaker_values[0]
    speaker_max = speaker_values[1]
    speaker_pivot = speaker_pivot.query(
        "@speaker_min<=`sentiment average`<=@speaker_max")
    st.write("Average sentiment range: ", speaker_values)
    # ----------------------

    st.dataframe(speaker_pivot.style.background_gradient(cmap=cm))

    fig, ax = plt.subplots()
    fig = px.scatter(speaker_pivot, x="sentiment average", y="sentiment sentence_count",
                     color="sentiment average", size="sentiment sentence_count", hover_data=['speaker'],
                     color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig)

    # Session Data
    session_pivot = make_pivot_table(df, ['session_title'], value_columns=['sentiment'],
                                     agg_function=(np.average, sentence_count))
    st.markdown("## **Sentiments by Session**")
    st.write("Data aggregated to show average sentiment and total sentences spoken for each session")

    # Interactive widgets----------------------
    session_values = st.slider(
        'Select a sentiment range',
        0.0, 1.0, (0.0, 1.0), key="session")
    session_min = session_values[0]
    session_max = session_values[1]
    session_pivot = session_pivot.query(
        "@session_min<=`sentiment average`<=@session_max")
    st.write("Average sentiment range: ", session_values)
    # ----------------------

    st.dataframe(session_pivot.style.background_gradient(cmap=cm))

    # fig, ax = plt.subplots()
    # fig = px.scatter(session_pivot, x="sentiment average", y="sentiment sentence_count",
    #                  color="sentiment average", hover_data=['session_title'],
    #                  color_continuous_scale=px.colors.sequential.Viridis)
    # st.plotly_chart(fig)

    st.markdown("## **Live Inference**")
    st.markdown("### Try out our Sentiment Analysis Model `xlm-roberta-base-sst-2-sh-sentiment`")

    sentiment_text = """‘Green shoots’ of recovery for Singapore’s economy, although uncertainties remain: Economists
    """

    text_box = st.text_area("Enter some text for Sentiment Analysis", sentiment_text)

    if text_box:
        url = 'https://hansard-nlp-api-l6lhxur2aq-uc.a.run.app/sentiment/'
        req_body = json.dumps({'hansard_text': text_box, 'output': {}})
        response = requests.post(url, data=req_body)
        response = json.loads(response.text)['output']

        st.write(response)
