from spacy import displacy
# from transformers import *
import spacy_streamlit
import streamlit as st
import spacy
from spacy_streamlit import visualize_ner
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px

default_text = """
Mr Louis Ng Kok Kwang asked the Minister for Health whether couples who have pre-implantation genetically screened embryos stored overseas can have their embryos shipped to Singapore given current travel restrictions during the pandemic.
The Parliamentary Secretary to the Minister for Health (Ms Rahayu Mahzam) (for the Minister for Health): Happy International Women's Day to all! During the pandemic, MOH received appeals from some couples to import their pre-implantation genetically screened embryos stored overseas. 

In reviewing each appeal, the Ministry considered whether processes and standards employed by overseas assisted reproduction (AR) centres are aligned to Singapore’s regulatory requirements under the Licensing Terms and Conditions for AR Services (AR LTCs). The Ministry may on an exceptional basis allow importation of the embryos, subject to conditions. These conditions include: (a) declaration by the overseas AR centre that the relevant requirements under the AR LTCs, including the handling, processing and storage of the embryos, are adhered to; (b) that no other findings besides the presence or absence of chromosomal aberrations are reported, and (c) proper documentation of the screening test results that were provided to the patient and attending physician in our local AR centres. 

Local AR centres which receive the tested embryos must also continue to ensure compliance with the AR LTCs.

Mr Deputy Speaker: Mr Louis Ng.

Mr Louis Ng Kok Kwang (Nee Soon): Thank you, Sir. I thank the Parliamentary Secretary for the reply. Could I ask whether we can make this more of a standard application? So, rather than an appeal and on exceptional basis, could we just have an application form during this pandemic where the couples cannot travel overseas to do their IVF during this period, can we have an application form where they can fill in to apply to transfer their embryos back to Singapore?

Ms Rahayu Mahzam: I thank the Member for the clarification. Typically, the applicants or those who are asking for this will usually just write in to appeal to MOH and we would provide them the answer and tell them what the necessary requirements are. But we can look into the suggestion and see how this information can be made more accessible, and perhaps, a form that is simplified for the purposes of this application. 

"""
st.title("Singapore Parliament NLP")
if st.checkbox('Show NER Demo'):
    model = st.sidebar.selectbox(
        'Model',
        (["en_core_web_sm"]))

    text_box = st.text_area("Enter some text for NER", default_text) 
    nlp = spacy.load(model)

    if text_box:
        doc = nlp(text_box)
        visualize_ner(doc, labels=nlp.get_pipe("ner").labels)


# sequence = """ Mr Louis Ng Kok Kwang asked the Minister for Health whether couples who have pre-implantation genetically screened embryos stored overseas can have their embryos shipped to Singapore given current travel restrictions during the pandemic.
# The Parliamentary Secretary to the Minister for Health (Ms Rahayu Mahzam) (for the Minister for Health): Happy International Women's Day to all! During the pandemic, MOH received appeals from some couples to import their pre-implantation genetically screened embryos stored overseas. 

# In reviewing each appeal, the Ministry considered whether processes and standards employed by overseas assisted reproduction (AR) centres are aligned to Singapore’s regulatory requirements under the Licensing Terms and Conditions for AR Services (AR LTCs). The Ministry may on an exceptional basis allow importation of the embryos, subject to conditions. These conditions include: (a) declaration by the overseas AR centre that the relevant requirements under the AR LTCs, including the handling, processing and storage of the embryos, are adhered to; (b) that no other findings besides the presence or absence of chromosomal aberrations are reported, and (c) proper documentation of the screening test results that were provided to the patient and attending physician in our local AR centres. 

# Local AR centres which receive the tested embryos must also continue to ensure compliance with the AR LTCs.

# Mr Deputy Speaker: Mr Louis Ng.

# Mr Louis Ng Kok Kwang (Nee Soon): Thank you, Sir. I thank the Parliamentary Secretary for the reply. Could I ask whether we can make this more of a standard application? So, rather than an appeal and on exceptional basis, could we just have an application form during this pandemic where the couples cannot travel overseas to do their IVF during this period, can we have an application form where they can fill in to apply to transfer their embryos back to Singapore?

# Ms Rahayu Mahzam: I thank the Member for the clarification. Typically, the applicants or those who are asking for this will usually just write in to appeal to MOH and we would provide them the answer and tell them what the necessary requirements are. But we can look into the suggestion and see how this information can be made more accessible, and perhaps, a form that is simplified for the purposes of this application. """

# st.title("Parliament NLP")

# text_box = st.text_area("Enter some text for NER", sequence) 

# HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
# model = XLMRobertaForTokenClassification.from_pretrained(
#     'asahi417/tner-xlm-roberta-base-ontonotes5')
# ner = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

# if text_box:

#     entity_list = ner(text_box)
#     for idx in entity_list:
#         idx['label'] = idx.pop('entity_group')

#     doc = {"text": text_box, "ents": entity_list}

#     html = displacy.render(doc, style="ent", manual=True)

#     html = html.replace("\n\n", "\n")
#     st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


@st.cache
def get_data():
    df = pd.read_csv("./Combined Data -14th-xlm-roberta-base-sst-2-handeset.csv")
    return df


@st.cache
def make_pivot_table(df, index_columns, value_columns, agg_function=np.sum):
    """ creates a pivot table for the given daraframe
    input: df = dataframe to make the pivot table out of
           index_columns = single column or set of two columns to be used as
                           columns to group by
           value_columns = the columns to aggregate as a target in the pivot table
           agg_function = what to use for the aggregation; default=np.sum
    output: pivot_color = pivot table with color scheme based on values
    """
    # create pivot table with internal pandas function
    pivot = pd.pivot_table(
        data=df,
        index=index_columns,
        values=value_columns,
        aggfunc= agg_function
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


st.markdown("## Sentiment Analysis")
cm = sn.light_palette("green", as_cmap=True)

# All sentiment data
df = get_data()
cols = df.columns.tolist()
st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)
st.dataframe(df[st_ms])

# Speaker Data
speaker_pivot = make_pivot_table(df, ['speaker'], value_columns=['sentiment'], 
                                 agg_function=(np.average, sentence_count))
st.markdown("### **Sentiments by Speaker**")
st.write("Data aggregated to show average sentiment and total sentences spoken for each session")

# Interactive widgets----------------------
speaker_values = st.slider(
    'Select a sentiment range',
    0.0, 1.0, (0.0, 1.0), key="speaker")
speaker_min = speaker_values[0]
speaker_max = speaker_values[1]
speaker_pivot = speaker_pivot.query("@speaker_min<=`sentiment average`<=@speaker_max")
st.write("Average sentiment range: ", speaker_values)
# ----------------------

st.dataframe(speaker_pivot.style.background_gradient(cmap=cm))

fig, ax = plt.subplots()
fig = px.scatter(speaker_pivot, x="sentiment average", y="sentiment sentence_count",
                 color="sentiment average", size= "sentiment sentence_count", hover_data=['speaker'],
                 color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig)


# Session Data
session_pivot = make_pivot_table(df, ['session_title'], value_columns=['sentiment'], 
                                 agg_function=(np.average, sentence_count))
st.markdown("### **Sentiments by Session**")
st.write("Data aggregated to show average sentiment and total sentences spoken for each session")

# Interactive widgets----------------------
session_values = st.slider(
    'Select a sentiment range',
    0.0, 1.0, (0.0, 1.0), key="session")
session_min = session_values[0]
session_max = session_values[1]
session_pivot = session_pivot.query("@session_min<=`sentiment average`<=@session_max")
st.write("Average sentiment range: ", session_values)
# ----------------------
st.dataframe(session_pivot.style.background_gradient(cmap=cm))

fig, ax = plt.subplots()
fig = px.scatter(session_pivot, x="sentiment average", y="sentiment sentence_count",
                 color="sentiment average", hover_data=['session_title'], 
                 color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig)
