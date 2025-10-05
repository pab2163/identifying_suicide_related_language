# Import standard libraries
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

# List of IDs
ids_validation = [ID_LIST]


# Base path
base_path = "/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/KeyInput"

# Output directory
output_dir = '../model_outputs_step1/low_lexicon_outputs_spellchecked/'
os.makedirs(output_dir, exist_ok=True)

# Iterate over subjects (spell-corrected data)
for id in ids_validation:
    filepath=''
    # One ID needs to be loaded from a different file to avoid non-utf8 encoding issues
    if id != id_need_recoding:
        filepath = os.path.join(base_path, str(id), f"corrected_{id}.csv")
    elif id == id_need_recoding:
        filepath = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/Preprocessed/spell_correct/spell_correctedgpt/id_need_recoding_full_with_pred_old.csv'
    print(f"Processing subject: {filepath}")
    input_path = filepath
    try:
        input_df = robust_read_csv(input_path)
    except FileNotFoundError:
        warnings.warn(f"File not found for subject {filepath}:. Skipping.")
        continue
    except Exception as e:
        warnings.warn(f"Error reading file for subject {filepath}: {e}. Skipping.")
        continue

    if 'subjectID' not in input_df.columns or 'corrected_message' not in input_df.columns:
        warnings.warn(f"Missing required columns in file for subject {filepath}. Skipping.")
        continue

    
    text_inputs = list(input_df.corrected_message.astype(str))

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
    counts.to_csv(f'{output_dir}{id}_srl_lexicon_counts.csv', index=False)

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
    features.to_csv(f'{output_dir}{id}_srl_lexicon_cts_features_sentence.csv', index=False)
