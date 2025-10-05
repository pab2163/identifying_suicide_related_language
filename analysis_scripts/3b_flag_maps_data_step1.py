import numpy as np
from lexicon_functions import *
import os

'''
Paul Alexander bloom
Run 5% of MAPS Data through Youth Suicide Lexicon and Swaminathan 2023 Lexicon
With spell correction
'''


# List of IDs
ids_validation = [ID_list]

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

# Pull in separate file for id_utf8_recode to avoid non-utf8 issues
spellid_utf8_recode = robust_read_csv('/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/Preprocessed/spell_correct/spell_correctedgpt/id_utf8_recode_full_with_pred_old.csv')

combined_df_spell = pd.concat([combined_df_spell, spellid_utf8_recode], ignore_index=True)

# youth suicide lexicon
youth_lexicon_results_spell = flag_lexicon_custom(input_df=combined_df_spell, text_column='corrected_message')
youth_lexicon_results_spell = flag_suicide_related_emojis(df=youth_lexicon_results_spell, text_column = 'corrected_message')

# swaminathan 2023 lexicon
youth_lexicon_results_spell = flag_lexicon_swaminathan_2023(input_df=youth_lexicon_results_spell, text_column='corrected_message')
youth_lexicon_results_spell.to_csv('../model_outputs_step1/maps_youth_lexicon_outputs_spellchecked.csv', index=False)


'''
Preprocessed - no spell correction
'''
input_df = pd.read_csv('../../../data/manual_coding/suicide_language/step_one_coding/coded_reconverted_csvs/final/final_codings.csv')

# youth suicide lexicon
youth_lexicon_results = flag_lexicon_custom(input_df=input_df, text_column='text_clean')
youth_lexicon_results = flag_suicide_related_emojis(df=youth_lexicon_results, text_column = 'text_clean')

# swaminathan 2023 lexicon
youth_lexicon_results = flag_lexicon_swaminathan_2023(input_df=youth_lexicon_results, text_column='text_clean')
youth_lexicon_results.to_csv('../model_outputs_step1/maps_youth_lexicon_outputs.csv', index=False)
