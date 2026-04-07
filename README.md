# Identifying Suicide-Related Language in Smartphone Keyboard Entries Among High-Risk Adolescents

Code for analyses for Bloom & Treves et. al., 2025 ([Preprint Link](https://osf.io/preprints/psyarxiv/gfa7h_v1))



## Part 1: Model Validation in Identifying Suicide-Related Language


### Analysis Scripts

All analysis and data-processing scripts are located in the [`analysis_scripts/`](analysis_scripts/) directory.  
The left column links directly to each script.

| Script | Description |
|---|---|
| [0_setup_local_embeddings.py](analysis_scripts/0_setup_local_embeddings.py) | Setup for local embeddings for calculating simlarity using the Low 2024 lexicon |
| [1_pull_construct_similarity.py](analysis_scripts/1_pull_construct_similarity.py) | Pull similarity in MAPS validation data to constructs in the Low 2024 Lexicon |
| [1b_pull_construct_similarity_predict.py](analysis_scripts/1b_pull_construct_similarity_predict.py) | Pull similarity in PREDICT validation data to constructs in the Low 2024 Lexicon |
| [2_assemble_data_to_code_step1.Rmd](analysis_scripts/2_assemble_data_to_code_step1.Rmd) | Create deidentified datasets for human coders of MAPS validation data (5% of participants) |
| [3a_aggregate_stage1_coding.Rmd](analysis_scripts/3a_aggregate_stage1_coding.Rmd) | Aggregate completed human coding of  MAPS and PREDICT validation data (N=22)  |
| [3b_flag_maps_data_step1.py](analysis_scripts/3b_flag_maps_data_step1.py) | Use Youth Suicide Lexicon and Swaminathan 2023 Lexicon to flag text entries in MAPS validation data (5% of participants) |
| [4a_merge_data_for_validation.Rmd](analysis_scripts/4a_merge_data_for_validation.Rmd) | Prep MAPS, PREDICT, and PED-SI validation data for calculating performance statistics |
| [4b_validation_step1_main.Rmd](analysis_scripts/4b_validation_step1_main.Rmd) | Calculate model performance statistics for identifying presence of suicide-related text in MAPS, PREDICT, and PED-SI validation data  |
| [4c_by_participant_validation.Rmd](analysis_scripts/4c_by_participant_validation.Rmd) | Calculate model performance stats separately by participant in MAPS validation data  |
| [helper_functions.R](analysis_scripts/helper_functions.R) | A variety of R helper functions for data analysis and cleaning |
| [lexicon_functions.py](analysis_scripts/lexicon_functions.py) | Functions for using the Youth Suicide Lexicon to flag data, as well as options for flagging emojis and using the Swaminathan 2023 Lexicon to flag data|


---


## Part 2: Examining Links Between Flagged Suicide-Related Language and Youth Suicidal Thoughts and Behaviors

| Script | Description |
|---|---|
| [5a_flag_maps_data_for_step2_coding.py](analysis_scripts/5a_flag_maps_data_for_step2_coding.py) | Use the Youth Suicide Lexicon to flag all text entries for MAPS and PREDICT cohorts (without spell-correction) to identify entries for step 2 human coding |
| [5b_delegate_step2.Rmd](analysis_scripts/5b_delegate_step2.Rmd) | Delegate human coders (for content category) for MAPS data |
| [5c_flag_maps_and_predict.py](analysis_scripts/5c_flag_maps_and_predict.py) | Use the Youth Suicide Lexicon to flag all text entries for MAPS and PREDICT cohorts (with and without spell-correction) for full analyses of lexicon flagging |
| [6a_maps_predict_stb_analysis_main.Rmd](analysis_scripts/6a_maps_predict_stb_analysis_main.Rmd) | Analyze patterns of flagged text in the MAPS and PREDICT cohorts as a function of lifetime STB history, baseline suicidal ideation, time of day, and app type |
| [6b_maps_predict_stb_analysis_nospellcorrect.Rmd](analysis_scripts/6b_maps_predict_stb_analysis_nospellcorrect.Rmd) | Parallel to 6a, but using the text flagging pipeline without spell correction |
| [6c_maps_predict_stb_analysis_no_exclusions.Rmd](analysis_scripts/6c_maps_predict_stb_analysis_no_exclusions.Rmd) | Parallel to 6a, but includes all available data (does not exclude participants with <1000 entries) |
| [6d_maps_predict_stb_analysis_multiverse.Rmd](analysis_scripts/6d_maps_predict_stb_analysis_multiverse.Rmd) | Multiverse sensitivity analyses of 6a analyses as a function of lifetime STB and baseline SI |
| [6e_maps_predict_stb_analysis_nbinom.Rmd](analysis_scripts/6e_maps_predict_stb_analysis_nbinom.Rmd) | Parallel to 6a, but uses negative binomial regression for daily rates of flagged language |
| [6f_sociodemographic_differences.Rmd](analysis_scripts/6f_sociodemographic_differences.Rmd) | Analysis of differences in frequency of flagged text via the youth suicide lexicon as a function of demographic variables |
| [7a_compile_step2_coding.Rmd](analysis_scripts/7a_compile_step2_coding.Rmd) | Compile human content codings of flagged entries in the MAPS cohort |
| [7b_step2_coding_descriptives.Rmd](analysis_scripts/7b_step2_coding_descriptives.Rmd) | Descrriptive analyses of content category frequency among entries flagged by the Youth Suicide Lexicon in the MAPS cohort |
| [8a_maps_coded_analyses.Rmd](analysis_scripts/8a_maps_coded_analyses.Rmd) | Analyses of human content labels as a function of lifetime STB history and baseline SI |
| [8b_maps_human_vs_lexicon.Rmd](analysis_scripts/8b_maps_human_vs_lexicon.Rmd) | Comparing the relative effect sizes of human coding versus lexicon flagging of text alone in associations with STB history or baseline SI |
| [8c_lexicon_plus_llm_analyses_stage2.Rmd](analysis_scripts/8c_lexicon_plus_llm_analyses_stage2.Rmd) | Comparing the relative effect sizes of a two-stage lexicon+LLM approach versus lexicon flagging of text alone in associations with STB history or baseline SI |
| [8d_lexicon_gpt_bootstrap.Rmd](analysis_scripts/8d_lexicon_gpt_bootstrap.Rmd) | Comparing the relative effect sizes of lexicon+LLM approaches versus lexicon flagging of text alone in associations with STB history or baseline SI |

### GPT Scripts

Scripts for using an institutional HIPAA-compliant version of GPT to identify suicide-related language

| Script | Description |
|---|---|
| [0_setup_for_stage1_inference.py](analysis_scripts/gpt_scripts/0_setup_for_stage1_inference.py) | Set up batch processing jsonl files for labeling of binary suicide language presence |
| [1_run_stage1_inference_batches.py](analysis_scripts/gpt_scripts/1_run_stage1_inference_batches.py) | Run batch processing of inference on binary suicide language presence (few-shot) |
| [2_retrieve_stage1_batches.py](analysis_scripts/gpt_scripts/2_retrieve_stage1_batches.py) | Retrieve batch processing GPT outputs |
| [3a_prep_context_window.Rmd](analysis_scripts/gpt_scripts/3a_prep_context_window.Rmd) | Set up MAPS & PREDICT text data for GPT processing using a structure of a focal text entry (e.g., flagged by the lexicon) plus surrounding text entries for context |
| [3b_setup_for_stage2_window_inference.py](analysis_scripts/gpt_scripts/3b_setup_for_stage2_window_inference.py) | Set up batch processing jsonl files for labeling of authentic, first-person, current suicide-related language |
| [3c_run_stage2_inference_batches_window.py](analysis_scripts/gpt_scripts/3c_run_stage2_inference_batches_window.py) | Run batch processing of infereence on classifying authentic, first-person, current suicide-related language (few-shot)  |
| [gpt_helper_functions.py](analysis_scripts/gpt_scripts/gpt_helper_functions.py) | Helper functions for GPT analysis pipelines |

## Youth Suicide Lexicon Files

Lexicon resources are stored in the [`analysis_scripts/lexicon_data/`](analysis_scripts/lexicon_data/) directory and related helper functions are in [`lexicon_functions.py`](analysis_scripts/lexicon_functions.py).

More flexible code for using the lexicon is available at: https://github.com/pab2163/youth_suicide_lexicon

## Lower-Level Keyboard Input Preprocessing Code:

Code & matrials for keyboard input preprocessing can be found in a separate github repository [here.](https://github.com/pab2163/auerbach_nlp)