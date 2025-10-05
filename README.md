# Identifying Suicide-Related Language in Smartphone Keyboard Entries Among High-Risk Adolescents

Code for analyses for Bloom & Treves et. al., 2025 ([Preprint Link](https://osf.io/preprints/psyarxiv/gfa7h_v1))



## Part 1: Youth Suicide Lexicon Development & Validation


## Analysis Scripts

All analysis and data-processing scripts are located in the [`analysis_scripts/`](analysis_scripts/) directory.  
The left column links directly to each script; descriptions can be added or edited as needed.

| Script | Description |
|---|---|
| [0_setup_local_embeddings.py](analysis_scripts/0_setup_local_embeddings.py) | Setup for local embeddings for calculating simlarity using the Low 2024 lexicon |
| [1_pull_construct_similarity.py](analysis_scripts/1_pull_construct_similarity.py) | Pull similarity in MAPS validation data to constructs in the Low 2024 Lexicon |
| [2_assemble_data_to_code_step1.Rmd](analysis_scripts/2_assemble_data_to_code_step1.Rmd) | Create deidentified datasets for human coders of MAPS validation data (5% of participants) |
| [3a_aggregate_stage1_coding.Rmd](analysis_scripts/3a_aggregate_stage1_coding.Rmd) | Aggregate completed human coding of MAPS validation data (5% of participants) |
| [3b_flag_maps_data_step1.py](analysis_scripts/3b_flag_maps_data_step1.py) | Use Youth Suicide Lexicon and Swaminathan 2023 Lexicon to flag text entries in MAPS validation data (5% of participants) |
| [4a_merge_data_for_validation.Rmd](analysis_scripts/4a_merge_data_for_validation.Rmd) | Prep MAPS and PED-SI validation data for calculating performance statistics |
| [4b_validation_step1.Rmd](analysis_scripts/4b_validation_step1.Rmd) | Calculate model performance statistics for identifying presence of suicide-related text in MAPS and PED-SI validation data  |
| [4c_by_participant_validation.Rmd](analysis_scripts/4c_by_participant_validation.Rmd) | Calculate model performance stats separately by participant in MAPS validation data  |
| [helper_functions.R](analysis_scripts/helper_functions.R) | A variety of R helper functions for data analysis and cleaning |
| [lexicon_functions.py](analysis_scripts/lexicon_functions.py) | Functions for using the Youth Suicide Lexicon to flag data, as well as options for flagging emojis and using the Swaminathan 2023 Lexicon to flag data|

---


## Stage 2: Examining Text Flagging by the Youth Suicide Lexicon In 2 Independent Youth Cohorts

| Script | Description |
|---|---|
| [5a_flag_maps_data_for_step2_coding.py](analysis_scripts/5a_flag_maps_data_for_step2_coding.py) | Use the Youth Suicide Lexicon to flag all text entries for MAPS and PREDICT cohorts (without spell-correction) to identify entries for step 2 human coding |
| [5b_delegate_step2.Rmd](analysis_scripts/5b_delegate_step2.Rmd) | Delegate human coders (for content category) for MAPS data |
| [5c_flag_maps_and_predict.py](analysis_scripts/5c_flag_maps_and_predict.py) | Use the Youth Suicide Lexicon to flag all text entries for MAPS and PREDICT cohorts (with and without spell-correction) for full analyses of lexicon flagging |
| [6a_maps_predict_stb_analysis.Rmd](analysis_scripts/6a_maps_predict_stb_analysis.Rmd) | Analyze patterns of flagged text in the MAPS and PREDICT cohorts as a function of lifetime STB history, baseline suicidal ideation, time of day, and app type |
| [6b_maps_predict_stb_analysis_nospellcorrect.Rmd](analysis_scripts/6b_maps_predict_stb_analysis_nospellcorrect.Rmd) | Parallel to 6a, but using the text flagging pipeline without spell correction |
| [6c_maps_predict_stb_analysis_no_exclusions.Rmd](analysis_scripts/6c_maps_predict_stb_analysis_no_exclusions.Rmd) | Parallel to 6a, but includes all available data (does not exclude participants with <1000 entries) |
| [6d_maps_predict_stb_analysis_multiverse.Rmd](analysis_scripts/6d_maps_predict_stb_analysis_multiverse.Rmd) | Multiverse sensitivity analyses of 6a analyses as a function of lifetime STB and baseline SI |
| [7a_compile_step2_coding.Rmd](analysis_scripts/7a_compile_step2_coding.Rmd) | Compile human content codings of flagged entries in the MAPS cohort |
| [7b_step2_coding_descriptives.Rmd](analysis_scripts/7b_step2_coding_descriptives.Rmd) | Descrriptive analyses of content category frequency among entries flagged by the Youth Suicide Lexicon in the MAPS cohort |
| [8a_maps_coded_analyses.Rmd](analysis_scripts/8a_maps_coded_analyses.Rmd) | Analyses of human content labels as a function of lifetime STB history and baseline SI |
| [8b_maps_human_vs_lexicon.Rmd](analysis_scripts/8b_maps_human_vs_lexicon.Rmd) | Comparing the relative effect sizes of human coding versus lexicon flagging of text alone in associations with STB history or baseline SI |


## Youth Suicide Lexicon Files

Lexicon resources are stored in the [`analysis_scripts/lexicon_data/`](analysis_scripts/lexicon_data/) directory and related helper functions are in [`lexicon_functions.py`](analysis_scripts/lexicon_functions.py).

More flexible code for using the lexicon is available at: https://github.com/pab2163/youth_suicide_lexicon

