import json
import os
import numpy as np
import re


def get_entity_count(folder_path):
    entity_count = {}

    for filename in os.listdir(folder_path):
        print(filename)

        with open(folder_path + "/" + filename) as json_file:
            data = json.load(json_file)

        for session in data['sessions']:
            session_sentiment = []
            parliament_sitting_date = re.findall(r"\d\d\d\d-\d\d-\d\d", data['filename'])[0] 
            for speech in session['speeches']:
                for content in speech['content']:

                    sentiment = content['sentiment']
                    session_sentiment.append(sentiment)

                    for entity in content['entities']:
                        label = entity['label']
                        if label in entity_count:
                            entity_count[label] += 1
                        else:
                            entity_count[label] = 0

    return entity_count


def get_speaker_entity_count(folder_path):
    speaker_entity_count = {}

    for filename in os.listdir(folder_path):
        print(filename)

        with open(folder_path + "/" + filename) as json_file:
            data = json.load(json_file)

        for session in data['sessions']:
            session_sentiment = []
            parliament_sitting_date = re.findall(r"\d\d\d\d-\d\d-\d\d", data['filename'])[0] 
            for speech in session['speeches']:
                name = speech['speaker']

                if name not in speaker_entity_count:
                    speaker_entity_count[name] = {}

                for content in speech['content']:
                    for entity in content['entities']:
                        label = entity['label']
                        if label in speaker_entity_count[name]:
                            speaker_entity_count[name][label] += 1
                        else:
                            speaker_entity_count[name][label] = 0

    return speaker_entity_count
