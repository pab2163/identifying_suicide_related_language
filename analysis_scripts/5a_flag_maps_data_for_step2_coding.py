import numpy as np
import os
import pandas as pd
from pathlib import Path
import datetime
from lexicon_functions import *


'''
Flag suicide language in ENTIRE MAPS dataset

Run both on the RAW data and Preprocessed data (although outputs will only show the preprocessed version to anonymize to the extent possible)
Output colums per participant for both raw and preprocessed

'''

def merge_text_sources(subfolder):
    print(f'Merging text sources for {subfolder}')
    try:
        files = {
            'raw': list(subfolder.glob('*KeyInput_message*.csv')),
            'preproc': list(subfolder.glob('Preprocessed*.csv'))
        }

        if not all(files.values()):
            missing = [k for k, v in files.items() if not v]
            raise FileNotFoundError(f"Missing files for: {', '.join(missing)}")

        # Robust read should be ok for emojis etc (unlike latin-1 encoding)
        df_raw = robust_read_csv(files['raw'][0]).drop_duplicates()
        df_preproc = robust_read_csv(files['preproc'][0]).drop_duplicates()
        df_preproc = df_preproc.drop('strinput_text', axis=1)

        merge_cols = ['id_app', 'tm_message_start', 'tm_message_end']

        df_merged = pd.merge(
            df_preproc,
            df_raw[merge_cols + ['id_message', 'id_row', 'strinput_text']],
            on=merge_cols,
            how='left'
        )

        df_merged.rename(columns={'text_clean': 'text_preproc', 'strinput_text': 'text_raw'}, inplace=True)

        return df_merged

    except Exception as e:
        raise RuntimeError(f"Error merging files in {subfolder.name}: {e}")

def batch_flag_suicide_language(input_dir, output_dir, debug=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for subfolder in input_dir.iterdir():
        if subfolder.is_dir():
            id_ = subfolder.name
            try:
                output_path = output_dir / f'{id_}_srl_flagged.csv'

                if not os.path.isfile(output_path):
                    df = merge_text_sources(subfolder)

                    merge_cols = ['id_app', 'id_message', 'id_row', 'tm_message_start', 'tm_message_end', 'cat_tz']

                    # Define columns to suffix
                    target_cols = [
                        'suicide_lexicon_custom_token',
                        'celeb_historical_suicide_token',
                        'suicide_lexicon_custom_pairs',
                        'suicide_lexicon_custom_full',
                        'emoji_flag',
                        'emojis_found'
                    ]

                    # Run and suffix raw flags (cumulatively)
                    raw_flags = flag_lexicon_custom(df, text_column='text_raw')
                    raw_flags = flag_suicide_related_emojis(raw_flags, text_column='text_raw')
                    raw_flags = raw_flags.rename(columns={col: f"{col}_raw" for col in raw_flags.columns if col in target_cols})
                    raw_flags = pd.concat([df[merge_cols], raw_flags[[f"{col}_raw" for col in target_cols if f"{col}_raw" in raw_flags.columns]]], axis=1)

                    # Run and suffix preproc flags (cumulatively)
                    preproc_flags = flag_lexicon_custom(df, text_column='text_preproc')
                    preproc_flags = flag_suicide_related_emojis(preproc_flags, text_column='text_preproc')
                    preproc_flags = preproc_flags.rename(columns={col: f"{col}_preproc" for col in preproc_flags.columns if col in target_cols})
                    preproc_flags = pd.concat([df[merge_cols], preproc_flags[[f"{col}_preproc" for col in target_cols if f"{col}_preproc" in preproc_flags.columns]]], axis=1)

                    # Merge flags to base df
                    result_df = df[merge_cols + ['text_preproc']]
                    result_df = result_df.merge(raw_flags, on=merge_cols, how='left')
                    result_df = result_df.merge(preproc_flags, on=merge_cols, how='left')

                    # Compute any_flagged from all flag columns
                    flag_cols = [col for col in result_df.columns if col.endswith('_raw') or col.endswith('_preproc')]
                    if 'text_preproc' in flag_cols:
                        flag_cols.remove('text_preproc')
                    result_df['any_flagged'] = result_df[flag_cols].apply(lambda row: any(row == 1), axis=1).astype(int)
                    n_flagged = int(result_df['any_flagged'].sum())
                    

                    result_df.to_csv(output_path, index=False)
                    results.append({'id': id_, 'status': 'success', 'error': None, 'n_flagged': n_flagged})
                else:
                    results.append({'id': id_, 'status': 'not run - file already present', 'error': None, 'n_flagged': None})

            except Exception as e:
                results.append({'id': id_, 'status': 'fail', 'error': str(e), 'n_flagged': None})
        if debug:
            break
    return pd.DataFrame(results)

output_dir=f'/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/srl_flagged_{str(datetime.datetime.today())}'

log_df = batch_flag_suicide_language(
    input_dir='/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/KeyInput',
    output_dir=output_dir,
    debug=False
)

log_df.to_csv(f'logs/srl_flag_log_{str(datetime.date.today())}.csv', index=False)
