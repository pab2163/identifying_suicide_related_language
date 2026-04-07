# Import standard libraries
'''
Flag PREDICT validation data using the Lexicon (with construct similarity) for N=11 PREDICT Participants

Paul Bloom

'''


import os
import sys
import shutil
import datetime
import copy
import warnings
import pandas as pd
import numpy as np
from lexicon_functions import *
from pathlib import Path
import glob 

# Append source directories to path
sys.path.append('./../src/')
sys.path.append('./../src/construct_tracker/')

# Import construct_tracker modules
from construct_tracker import lexicon
from construct_tracker import cts
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load Lexicons
# ---------------------------

try:
    # Load full SRL lexicon
    srl = lexicon.load_lexicon(name='srl_v1-0')

    # Load prototype tokens only
    srl_prototypes = lexicon.load_lexicon(name='srl_prototypes_v1-0')
except Exception as e:
    raise RuntimeError(f"Error loading lexicons: {e}")

# ---------------------------
# Process Subject Files
# ---------------------------

file_paths_predict = glob.glob('/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/manual_coding/suicide_language_prediction/predict_coding_stage1/blank/*.csv')

# Output directory
output_dir = '../model_outputs_step1/low_lexicon_outputs_spellchecked_predict/'
os.makedirs(output_dir, exist_ok=True)

# Iterate over subjects (spell-corrected data)
for filepath in file_paths_predict:
    print(f"Processing subject: {filepath}")
    filestring = Path(filepath).name
    try:
        input_df = robust_read_csv(filepath)
    except FileNotFoundError:
        warnings.warn(f"File not found for subject {filepath}:. Skipping.")
        continue
    except Exception as e:
        warnings.warn(f"Error reading file for subject {filepath}: {e}. Skipping.")
        continue

    if 'subjectID' not in input_df.columns or 'text_clean' not in input_df.columns:
        warnings.warn(f"Missing required columns in file for subject {filepath}. Skipping.")
        continue

    
    text_inputs = list(input_df.text_clean.astype(str))

    # ----------------------------------
    # Exact Lexicon Match Feature Extraction
    # ----------------------------------

    try:
        counts, matches_by_construct, matches_doc2construct, matches_construct2doc = srl.extract(
            text_inputs,
            normalize=False,
        )
    except Exception as e:
        warnings.warn(f"Error during lexicon extraction for subject {filepath}: {e}. Skipping.")
        continue

    counts = pd.concat([input_df, counts], axis=1)
    counts.to_csv(f'{output_dir}/count_{filestring}', index=False)

    # ----------------------------------
    # Cosine Similarity Features (Sentence-Level)
    # ----------------------------------

    # Convert lexicon to dict format
    lexicon_dict = {
        c: srl_prototypes.constructs[c]["tokens"]
        for c in srl_prototypes.constructs
    }

    if not text_inputs or all([not isinstance(t, str) or t.strip() == '' for t in text_inputs]):
        warnings.warn(f"No valid text inputs for subject {filepath}. Skipping.")
        continue

    if not lexicon_dict:
        warnings.warn(f"Lexicon dictionary is empty for subject {filepath}. Skipping.")
        continue

    try:
        features, lexicon_dict_final_order, cosine_similarities = cts.measure(
            lexicon_dict,
            text_inputs,
            count_if_exact_match=False,
            summary_stat=['max', 'mean'],
            embeddings_model='models/all-MiniLM-L6-v2-local',
            stored_embeddings_path='data/embeddings/stored_embeddings.pickle',
            save_lexicon_embeddings=True,
            verbose=True,
            document_representation="sentence"
        )

        if features is None or features.empty:
            warnings.warn(f"No similarity features computed for subject {filepath}. Skipping.")
            continue

    except Exception as e:
        warnings.warn(f"Error during cosine similarity computation for subject {filepath}: {e}. Skipping.")
        continue


    try:
        features, lexicon_dict_final_order, cosine_similarities = cts.measure(
            lexicon_dict,
            text_inputs,
            count_if_exact_match=False,
            summary_stat=['max', 'mean'],
            embeddings_model='models/all-MiniLM-L6-v2-local',
            stored_embeddings_path='data/embeddings/stored_embeddings.pickle',
            save_lexicon_embeddings=True,
            verbose=True,
            document_representation="sentence"
        )
    except Exception as e:
        warnings.warn(f"Error during cosine similarity computation for subject {filepath}: {e}. Skipping.")
        continue

    features = pd.concat([input_df, features], axis=1)
    features.to_csv(f'{output_dir}/similarity_{filestring}', index=False)
