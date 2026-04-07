'''
Setup batch job jsonl files for using GPT to identify binary presence vs. absence of suicide-related language
'''

import openai
import os
import pandas as pd
import json
import glob

from gpt_helper_functions import *

# initialize client with openai key to ensure use of the privacy-protected version
init_client("openai_key.txt")

model="gpt-4o-mini"

# System prompt for binary identification of presence of suicide-related language 
system_prompt_binary = """You are an expert in identifying if a piece of text references suicide.

You will be given a piece of text and you should output a json object with one field: "label".
The value of "label" should be "Yes" if the text references suicidal thoughts, behaviors, or methods — including indirect, euphemistic, hotlines (988, trevor project), or misspelled expressions. Output "No" otherwise.
Only output the json object and nothing else.

Examples:
Text: "thinking about sewerslide, just wanna kms"
Label: {"label": "Yes"}

Text: "how much advil will overdose take"
Label: {"label": "Yes"}

Text: "just feeling so tired of life"
Label: {"label": "Yes"}

Text: "he treats me more like a crisis line than his boyfriend"
Label: {"label": "Yes"}

Text: "its been 3 years since i wanted to kill myself"
Label: {"label": "Yes"}

Text: "im depressed today"
Label: {"label": "No"}

Text: "my phone is gonna die"
Label: {"label": "No"}

Text: "lets go kill that guy"
Label: {"label": "No"}
"""

# MAPS

# List of IDs
with open('../maps_validation_ids.txt', 'r') as file:
    ids_validation = file.readlines()

# Base path
base_path = "/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/KeyInput"

file_paths = ['']

# Loop through IDs and read in each CSV
for subject_id in ids_validation:
    file_paths.append(os.path.join(base_path, str(subject_id), f"corrected_final_{subject_id}.csv"))


output_dir = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step1/gpt/fewshot_v1'

# make output dir if it does not exist already
os.makedirs(output_dir, exist_ok=True)


print(file_paths)

batch_csv_to_jsonl(csv_file_paths=file_paths, 
             output_dir=output_dir,
             text_column='corrected_message', 
             model=model, 
             system_message=system_prompt_binary,
             entry_id_column='entry_id',
             other_id_column='subjectID',
             reindex=True, 
             logfile = f'{output_dir}/batching_logfile.csv')


# PREDICT
file_paths_predict = glob.glob('/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/manual_coding/suicide_language_prediction/predict_coding_stage1/blank/*.csv')
output_dir_predict = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step1/gpt/fewshot_predict_v1'

batch_csv_to_jsonl(csv_file_paths=file_paths_predict, 
             output_dir=output_dir_predict,
             text_column='text_clean', 
             model=model, 
             system_message=system_prompt_binary,
             entry_id_column='entry_id',
             other_id_column='subjectID',
             reindex=False, 
             logfile = f'{output_dir_predict}/batching_logfile.csv')

print(file_paths_predict)