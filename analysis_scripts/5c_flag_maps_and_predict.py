import pandas as pd
from pathlib import Path
from lexicon_functions import *
import argparse
import glob
import datetime

'''
Paul Alexander Bloom
This code flags all data in the MAPS and PREDICT cohorts with the Youth Suicide Lexicon

Runs on preprocessed anonymized data both with and without spell-correction

'''


def flag_files_with_filepath(filepaths, output_path, text_column='text_clean', keep_columns=None):
    """
    Flags suicide-related language in a list of CSV files using lexicon-based methods.

    Each file is read, text in the specified column is cleaned and checked for UTF-8 compatibility,
    then various flagging functions are applied. The results are saved to a single combined output CSV,
    and an additional "notext" version of the file is saved with the text column removed.

    Parameters:
    - filepaths: list of str or Path
        List of paths to input CSV files to process.
    - output_path: str or Path
        Path to save the combined output CSV file with flags.
    - text_column: str
        Name of the column in each CSV that contains the text to flag.
    - keep_columns: list of str or None
        List of additional columns to retain from the original CSV (besides the text column).
        If None, only the text_column is retained before flagging.
    
    Returns:
    - pd.DataFrame: The combined flagged DataFrame.
    """
    results = []
    filepaths = [Path(fp) for fp in filepaths]
    for fp in filepaths:
        try:
            df = robust_read_csv(fp).drop_duplicates()

            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in file: {fp}")

            # Remove rows with non-UTF-8 encodable text
            def is_utf8_encodable(val):
                try:
                    val.encode('utf-8')
                    return True
                except Exception:
                    return False

            df = df[df[text_column].apply(lambda x: isinstance(x, str) and is_utf8_encodable(x))]

            # Select columns to keep
            base_cols = [text_column]
            if keep_columns:
                base_cols.extend([col for col in keep_columns if col in df.columns])
            df = df[base_cols].copy()

            # Run flaggers
            print(f'\n\nExtracting suicide lexicon tokens in {fp}')
            df_flags = flag_lexicon_custom(df, text_column=text_column)
            df_flags = flag_suicide_related_emojis(df_flags, text_column=text_column)
            df_flags = flag_lexicon_swaminathan_2023(df_flags, text_column=text_column, debug=False)
            df_flags['filepath'] = str(fp)

            results.append(df_flags)

        except Exception as e:
            print(f"Failed to process {fp}: {e}")

    if results:
        combined_df = pd.concat(results, ignore_index=True)

        # Save full version
        output_path = Path(output_path)
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Output saved to {output_path}")

        # Save notext version
        notext_path = output_path.with_name(output_path.stem + '_notext.csv')
        combined_df_notext = combined_df.drop(columns=[text_column], errors='ignore')
        combined_df_notext.to_csv(notext_path, index=False, encoding='utf-8')
        print(f"Text-removed version saved to {notext_path}")

        return combined_df
    else:
        print("No files processed.")
        return pd.DataFrame()


maps_dir='/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/KeyInput'
predict_dir='/Volumes/AUERBACHLAB/Columbia/MAPS_Language/data/PREDICT'

maps_files=glob.glob(f'{maps_dir}/*/Preprocessed_2025_07_01.csv')
predict_files=glob.glob(f'{predict_dir}/*/Preprocessed_2025_07_29.csv')

# Partly preprocessed (sanitization, removal of duplicates, etc)
flag_files_with_filepath(filepaths=predict_files,
                         output_path = f'../data_stage2/predict_flagged_vs_maps_preproc_{datetime.datetime.now().date()}.csv',
                         text_column='text_clean',
                         keep_columns=['id_message', 
                                       'id_row',
                                       'tm_message_start', 
                                       'tm_message_end', 
                                       'id_app', 
                                       'cat_tz',
                                       'id_session'])

flag_files_with_filepath(filepaths=maps_files,
                         output_path = f'../data_stage2/maps_flagged_vs_predict_preproc_{datetime.datetime.now().date()}.csv',
                         text_column='text_clean',
                         keep_columns=['id_message', 
                                       'id_row',
                                       'tm_message_start', 
                                       'tm_message_end', 
                                       'id_app', 
                                       'cat_tz',
                                       'id_session'])

# Fully preprocessed - with translation/spell correction


maps_files_final=glob.glob(f'{maps_dir}/*/corrected_final_*.csv')
predict_files_final=glob.glob(f'{predict_dir}/*/*corrected_final_*.csv')

maps_files_final.sort()
predict_files_final.sort()
print(len(maps_files_final))
print(len(predict_files_final))

flag_files_with_filepath(filepaths=maps_files_final,
                         output_path = f'../data_stage2/maps_flagged_vs_predict_final_{datetime.datetime.now().date()}.csv',
                         text_column='corrected_message',
                         keep_columns=['id_message', 
                                       'id_row',
                                       'tm_message_start', 
                                       'tm_message_end', 
                                       'text_clean',
                                       'language_translate',
                                       'language',
                                       'id_app', 
                                       'cat_tz',
                                       'id_session'])


flag_files_with_filepath(filepaths=predict_files_final,
                         output_path = f'../data_stage2/predict_flagged_vs_maps_final_{datetime.datetime.now().date()}.csv',
                        text_column='corrected_message',
                         keep_columns=['id_message', 
                                       'id_row',
                                       'tm_message_start', 
                                       'tm_message_end', 
                                       'text_clean',
                                       'language_translate',
                                       'language',
                                       'id_app', 
                                       'cat_tz',
                                       'id_session'])





