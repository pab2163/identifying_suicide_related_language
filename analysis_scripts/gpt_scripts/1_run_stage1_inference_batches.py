'''
Launch GPT Batch Inference Jobs (Few-Shot) for Binary Suicide-Related Language Identification
'''

import openai
import os
import pandas as pd
import json
import glob
from gpt_helper_functions import *

# initialize client with openai key to ensure use of the privacy-protected version
init_client("openai_key.txt")

#MAPS
# the logfile (created in prior step, has information on which jsonls to feed into gpt)
output_dir = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step1/gpt/fewshot_v1'
logfile = f'{output_dir}/batching_logfile.csv'

# Inspect log to confirm JSONLs were created ────────────────────────────
log_df = pd.read_csv(logfile)
print("\nLog after JSONL creation:")
print(log_df[['input_jsonl', 'source_csv', 'jsonl_created_timestamp']].to_string())


# Launch batch jobs ──────────────────────────────────────────────────────
input_jsonls = log_df['input_jsonl'].tolist()
launch_multiple_batch_requests(
    jsonl_filepaths=input_jsonls,
    description='Stage 1 Few-Shot Pipeline',
    logfile=logfile
)

# Check on launched batch jobs
log_df = pd.read_csv(logfile)
print("\nLog after launch:")
print(log_df[['input_jsonl', 'batch_id', 'launch_timestamp']].to_string())


# PREDICT
# the logfile (created in prior step, has information on which jsonls to feed into gpt)
output_dir_predict = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step1/gpt/fewshot_predict_v1'
logfile = f'{output_dir_predict}/batching_logfile.csv'

# Inspect log to confirm JSONLs were created ────────────────────────────
log_df = pd.read_csv(logfile)
print("\nLog after JSONL creation:")
print(log_df[['input_jsonl', 'source_csv', 'jsonl_created_timestamp']].to_string())


# Launch batch jobs ──────────────────────────────────────────────────────
input_jsonls = log_df['input_jsonl'].tolist()
launch_multiple_batch_requests(
    jsonl_filepaths=input_jsonls,
    description='Stage 1 Few-Shot Pipeline',
    logfile=logfile
)

# Check on launched batch jobs
log_df = pd.read_csv(logfile)
print("\nLog after launch:")
print(log_df[['input_jsonl', 'batch_id', 'launch_timestamp']].to_string())