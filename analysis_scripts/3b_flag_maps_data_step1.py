
import numpy as np
from lexicon_functions import *
import os

'''
Flag 5% of MAPS Data (11 Participants)
'''

'''
With spell correction
'''


# List of IDs
ids_validation = "REMOVED_FOR_DEIDENTIFICATION"

# Base path
base_path = "/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/KeyInput"

# List to store individual dataframes
dfs = []

# Loop through IDs and read in each CSV
for subject_id in ids_validation:
    file_path = os.path.join(base_path, str(subject_id), f"corrected_{subject_id}.csv")
    df = robust_read_csv(file_path)
    df['subjectID'] = subject_id
    dfs.append(df)
    
# Concatenate all dataframes row-wise
combined_df_spell = pd.concat(dfs, ignore_index=True)


# custom lexicon
youth_lexicon_results_spell = flag_lexicon_custom(input_df=combined_df_spell, text_column='corrected_message')
youth_lexicon_results_spell = flag_suicide_related_emojis(df=youth_lexicon_results_spell, text_column = 'corrected_message')

# swaminathan 2023 lexicon
youth_lexicon_results_spell = flag_lexicon_swaminathan_2023(input_df=youth_lexicon_results_spell, text_column='corrected_message')
youth_lexicon_results_spell.to_csv('../model_outputs_step1/maps_youth_lexicon_outputs_spellchecked.csv', index=False)



'''
Preprocessed - no spell correction
'''
input_df = pd.read_csv('../../../data/manual_coding/suicide_language/step_one_coding/coded_reconverted_csvs/final/final_codings.csv')

# custom lexicon
youth_lexicon_results = flag_lexicon_custom(input_df=input_df, text_column='text_clean')
youth_lexicon_results = flag_suicide_related_emojis(df=youth_lexicon_results, text_column = 'text_clean')

# swaminathan 2023 lexicon
youth_lexicon_results = flag_lexicon_swaminathan_2023(input_df=youth_lexicon_results, text_column='text_clean')
youth_lexicon_results.to_csv('../model_outputs_step1/maps_youth_lexicon_outputs.csv', index=False)

''''
Raw data flagging
'''
raw_text = pd.read_csv('/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/Preprocessed/for_srl_head_to_head/raw.csv')

# custom lexicon
youth_lexicon_results = flag_lexicon_custom(input_df=raw_text, text_column='strinput_text')
youth_lexicon_results = flag_suicide_related_emojis(df=youth_lexicon_results, text_column = 'strinput_text')

# swaminathan 2023 lexicon
youth_lexicon_results = flag_lexicon_swaminathan_2023(input_df=raw_text, text_column='strinput_text')
youth_lexicon_results.to_csv('../model_outputs_step1/maps_youth_lexicon_outputs_raw.csv', index=False)

# youth_lexicon_results_redit.to_csv('../model_outputs_step1/redit_results.csv', index=False)