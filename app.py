import spacy_streamlit
import streamlit as st
import spacy
from spacy_streamlit import visualize_ner

default_text = """
Mr Louis Ng Kok Kwang asked the Minister for Health whether couples who have pre-implantation genetically screened embryos stored overseas can have their embryos shipped to Singapore given current travel restrictions during the pandemic.
The Parliamentary Secretary to the Minister for Health (Ms Rahayu Mahzam) (for the Minister for Health): Happy International Women's Day to all! During the pandemic, MOH received appeals from some couples to import their pre-implantation genetically screened embryos stored overseas. 

In reviewing each appeal, the Ministry considered whether processes and standards employed by overseas assisted reproduction (AR) centres are aligned to Singapore’s regulatory requirements under the Licensing Terms and Conditions for AR Services (AR LTCs). The Ministry may on an exceptional basis allow importation of the embryos, subject to conditions. These conditions include: (a) declaration by the overseas AR centre that the relevant requirements under the AR LTCs, including the handling, processing and storage of the embryos, are adhered to; (b) that no other findings besides the presence or absence of chromosomal aberrations are reported, and (c) proper documentation of the screening test results that were provided to the patient and attending physician in our local AR centres. 

Local AR centres which receive the tested embryos must also continue to ensure compliance with the AR LTCs.

Mr Deputy Speaker: Mr Louis Ng.

Mr Louis Ng Kok Kwang (Nee Soon): Thank you, Sir. I thank the Parliamentary Secretary for the reply. Could I ask whether we can make this more of a standard application? So, rather than an appeal and on exceptional basis, could we just have an application form during this pandemic where the couples cannot travel overseas to do their IVF during this period, can we have an application form where they can fill in to apply to transfer their embryos back to Singapore?

Ms Rahayu Mahzam: I thank the Member for the clarification. Typically, the applicants or those who are asking for this will usually just write in to appeal to MOH and we would provide them the answer and tell them what the necessary requirements are. But we can look into the suggestion and see how this information can be made more accessible, and perhaps, a form that is simplified for the purposes of this application. 

"""
st.title("Parliament NLP")
model = st.sidebar.selectbox(
    'Model',
    ("en_core_web_sm", "en_core_web_trf"))


text_box = st.text_area("Enter some text for NER", default_text) 
nlp = spacy.load(model)

if text_box:
    doc = nlp(text_box)
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

# from transformers import *
# import streamlit as st
# from spacy import displacy

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
