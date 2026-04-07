'''
Run batched GPT for few-shot inference for whether lexicon-flagged text is current, authentic, and first-person

'''

import openai
import os
import pandas as pd
import json
import glob
from gpt_helper_functions import *

# initialize client with openai key to ensure use of the privacy-protected version
init_client("openai_key.txt")

# the logfile (created in prior step, has information on which jsonls to feed into gpt)
output_dir = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step2/gpt/fewshot_window_predict_v2'
logfile = f'{output_dir}/batching_logfile.csv'

# Inspect log to confirm JSONLs were created ────────────────────────────
log_df = pd.read_csv(logfile)
print("\nLog after JSONL creation:")
print(log_df[['input_jsonl', 'source_csv', 'jsonl_created_timestamp']].to_string())


# Launch batch jobs ──────────────────────────────────────────────────────
input_jsonls = log_df['input_jsonl'].tolist()
launch_multiple_batch_requests(
    jsonl_filepaths=input_jsonls,
    description='Stage 2 Few-Shot Pipeline Window 2',
    logfile=logfile
)

# Check on launched batch jobs
log_df = pd.read_csv(logfile)
print("\nLog after launch:")
print(log_df[['input_jsonl', 'batch_id', 'launch_timestamp']].to_string())