''''
Retrieve batched jsonl outputs for GPT binary suicide-related language inference
'''

import openai
import os
import pandas as pd
import json
from gpt_helper_functions import *
import string
import glob

# initialize client with openai key to ensure use of the privacy-protected version
init_client("openai_key.txt")

# the logfile (created in prior step, has information on which jsonls to feed into gpt)
output_dir = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step1/gpt/fewshot_v1'
logfile = f'{output_dir}/batching_logfile.csv'
output_jsonl_dir = os.path.join(output_dir, 'output_jsonl')
results_dir = os.path.join(output_dir, 'results')

retrieve_all_batches(logfile=logfile, output_dir=output_jsonl_dir)

log_df = pd.read_csv(logfile)
print("\nLog after retrieval:")
print(log_df[['batch_id', 'retrieval_status', 'output_path']].to_string())

# Parse and join outputs to input CSVs ───────────────────────────────────

results = process_all_inputs(
    output_dir=results_dir,
    entry_id_column='entry_id',
    other_id_column='subjectID',
    logfile=logfile,
    expected_column='label'
)

print("\nOutput files:")
for input_csv, output_csv in results.items():
    print(f"  {input_csv} → {output_csv}")


# Make sure to clean the text identically to lexicon outputs for good joins
def preproc_text(text):
    """
    Preprocesses input text by converting it to lowercase and removing punctuation,
    except for the '|' character.

    Args:
        text (str): The input string to preprocess.

    Returns:
        str: The cleaned and lowercased text with '|' preserved.
    """
    allowed_punct = '|'
    punct_to_remove = ''.join(ch for ch in string.punctuation if ch not in allowed_punct)
    text = text.lower()
    text = text.translate(str.maketrans('', '', punct_to_remove))
    return text


output_csv_files = glob.glob(f'/{results_dir}/*.csv')

for csv_file in output_csv_files:
    df = pd.read_csv(csv_file)
    df['corrected_message'] = df['corrected_message'].astype(str)
    df['corrected_final'] = df['corrected_message'].apply(preproc_text)
    df.to_csv(csv_file, index=False)
    




