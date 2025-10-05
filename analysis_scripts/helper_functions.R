# ============================================================
# Libraries
# ============================================================

library(tidyverse)
library(caret)
library(ggplot2)
library(stringr)
library(cowplot)
library(dplyr)
library(purrr)


# ============================================================
# Function: evaluate_predictions()
# ============================================================
# Purpose:
#   Computes classification performance metrics for one or more
#   prediction columns in a dataframe, using caret::confusionMatrix.
#
# Arguments:
#   maps_validation_df : Data frame containing reference and prediction columns
#   reference_col       : Name of the ground-truth (reference) column
#   prediction_cols     : Character vector of prediction column names
#   positive_class      : Label for the positive class (default = '1')
#
# Returns:
#   A data frame of evaluation metrics (e.g., Precision, Recall, F1)
#   for each prediction column.
# ============================================================
evaluate_predictions = function(maps_validation_df, reference_col, prediction_cols, positive_class = '1') {
  
  # Check that the reference column exists
  if (!reference_col %in% names(maps_validation_df)) {
    stop(paste("Reference column '", reference_col, "' not found in the dataframe."))
  }
  
  # Check that all prediction columns exist
  missing_preds = setdiff(prediction_cols, names(maps_validation_df))
  if (length(missing_preds) > 0) {
    stop(paste("Prediction column(s) '", paste(missing_preds, collapse = ", "), "' not found in the dataframe."))
  }
  
  # Initialize list to store results
  results_list = list()
  
  # Loop through prediction columns
  for (pred_col in prediction_cols) {
    # Convert to factor for confusionMatrix
    predicted = factor(maps_validation_df[[pred_col]])
    reference = factor(maps_validation_df[[reference_col]])
    
    # Compute confusion matrix statistics
    cm = caret::confusionMatrix(
      data = predicted,
      reference = reference,
      positive = positive_class,
      mode = 'everything'
    )
    
    # Extract key metrics
    stats = data.frame(
      Sensitivity = cm$byClass[['Sensitivity']],
      Specificity = cm$byClass[['Specificity']],
      Pos_Pred_Value = cm$byClass[['Pos Pred Value']],
      Neg_Pred_Value = cm$byClass[['Neg Pred Value']],
      Precision = cm$byClass[['Precision']],
      Recall = cm$byClass[['Recall']],
      F1 = cm$byClass[['F1']],
      Prevalence = cm$byClass[['Prevalence']],
      Detection_Rate = cm$byClass[['Detection Rate']],
      Detection_Prevalence = cm$byClass[['Detection Prevalence']],
      Balanced_Accuracy = cm$byClass[['Balanced Accuracy']]
    )
    
    # Add prediction column name
    stats$PredictionColumn = pred_col
    
    # Store results
    results_list[[pred_col]] = stats
  }
  
  # Combine all results into a single data frame
  results_df = do.call(rbind, results_list)
  
  return(results_df)
}



# ============================================================
# Function: bootstrap_evaluate_predictions()
# ============================================================
# Purpose:
#   Performs bootstrapped evaluation of model predictions across
#   multiple resampled datasets to estimate variability.
#
# Arguments:
#   full_df          : Full dataset
#   reference_col    : Name of ground-truth column
#   prediction_cols  : Character vector of prediction column names
#   positive_class   : Label for positive class (default = '1')
#   n_iterations     : Number of bootstrap iterations (default = 100)
#   seed             : Random seed for reproducibility (default = 123)
#
# Returns:
#   Data frame of evaluation metrics for all bootstrap iterations.
# ============================================================
bootstrap_evaluate_predictions = function(
    full_df,
    reference_col,
    prediction_cols,
    positive_class = '1',
    n_iterations = 100,
    seed = 123
) {
  set.seed(seed)
  
  bootstrap_results = list()
  
  # Run bootstrap iterations
  for (i in 1:n_iterations) {
    # Bootstrap sample (sample with replacement)
    boot_sample = full_df[sample(nrow(full_df), replace = TRUE), ]
    
    # Evaluate predictions on the bootstrap sample
    eval_result = evaluate_predictions(
      boot_sample,
      reference_col = reference_col,
      prediction_cols = prediction_cols,
      positive_class = positive_class
    )
    
    # Add iteration ID
    eval_result$BootstrapIteration = i
    bootstrap_results[[i]] = eval_result
  }
  
  # Combine results into one long data frame
  results_df = do.call(rbind, bootstrap_results)
  
  # Reorder columns for clarity
  results_df = results_df[, c("BootstrapIteration", "PredictionColumn",
                              setdiff(names(results_df), c("BootstrapIteration", "PredictionColumn")))]
  
  return(results_df)
}



# ============================================================
# Function: sample_participants()
# ============================================================
# Purpose:
#   Performs bootstrapping by resampling participants (IDs)
#   with replacement, preserving their within-participant data.
#
# Arguments:
#   data : Data frame containing a column `ID` representing participants
#
# Returns:
#   A bootstrapped version of the dataset with re-labeled participant IDs.
# ============================================================
sample_participants = function(data) {
  
  ids = unique(data$ID)
  n_ids = length(ids)
  
  # Sample participant IDs with replacement
  sampled_ids = sample(ids, n_ids, replace = TRUE)
  
  # Build new dataset by stacking sampled participants
  counter = 1
  for (id in sampled_ids) {
    tmp = dplyr::filter(data, ID == id)
    tmp$ID = counter
    if (counter == 1) {
      out = tmp
    } else {
      out = rbind(out, tmp)
    }
    counter = counter + 1
  }
  
  return(out)
}



# ============================================================
# Function: combine_coder_files()
# ============================================================
# Purpose:
#   Combines multiple coder CSV files into a single data frame.
#   Each file’s unique coding column is renamed with the coder’s name.
#
# Arguments:
#   file_paths       : Character vector of file paths to coder CSV files
#   unique_col_name  : Column name that identifies the coder's unique variable
#
# Returns:
#   A merged data frame containing all coders’ data.
# ============================================================
combine_coder_files = function(file_paths, unique_col_name) {
  
  data_list = map(file_paths, function(path) {
    df = read_csv(path, show_col_types = FALSE)
    coder_name = tools::file_path_sans_ext(basename(path))
    
    # Validate that the target column exists
    if (!(unique_col_name %in% names(df))) {
      stop(glue::glue("Column `{unique_col_name}` not found in {path}"))
    }
    
    # Rename the unique coding column to the coder’s name
    df = df %>% rename(!!coder_name := all_of(unique_col_name))
    df
  })
  
  # Identify shared columns (to use as join keys)
  join_keys = reduce(data_list, function(x, y) intersect(names(x), names(y)))
  
  # Merge all coder data
  combined_df = reduce(data_list, full_join, by = join_keys)
  
  return(combined_df)
}



# ============================================================
# Function: coalesce_coder_columns()
# ============================================================
# Purpose:
#   Merges coding data from two coders into a tidy comparison format
#   based on a provided coder key.
#
# Arguments:
#   combined_df     : Data frame containing all coder data
#   coder_key_path  : Path to CSV file linking subjectIDs to coder IDs
#
# Returns:
#   A tidy data frame with both coders’ codes side by side.
# ============================================================
coalesce_coder_columns = function(combined_df, coder_key_path) {
  
  # Read in the coder key (subjectID, coder1, coder2)
  coder_key = read_csv(coder_key_path, show_col_types = FALSE)
  
  # Reshape combined data: one row per subjectID per coder
  long_df = combined_df %>%
    pivot_longer(
      cols = contains('step_one_coded'),
      names_to = "coder",
      values_to = "code"
    ) %>%
    mutate(coder = gsub('step_one_coded_', '', coder))
  
  # Merge coder1 data
  df_coder1 = coder_key %>%
    select(subjectID, coder = coder1) %>%
    left_join(long_df, by = c("subjectID", "coder")) %>%
    rename(code1 = code, coder1_name = coder)
  
  # Merge coder2 data
  df_coder2 = coder_key %>%
    select(subjectID, coder = coder2) %>%
    left_join(long_df, by = c("subjectID", "coder")) %>%
    rename(code2 = code, coder2_name = coder)
  
  # Combine coder1 and coder2 data by subject-level metadata
  final_df = df_coder1 %>%
    left_join(
      df_coder2,
      by = c('subjectID', 'tm_message_start', 'tm_message_end',
             'timeDiff', 'id_app', 'text_clean')
    )
  
  # Quick check: count per subjectID-coder pair
  check = final_df %>%
    group_by(subjectID, coder1_name, coder2_name) %>%
    count()
  
  print(check)
  
  # Reorder columns for readability
  final_df = final_df %>%
    select(-code1, -code2, -coder1_name, -coder2_name,
           everything(), code1, code2, coder1_name, coder2_name)
  
  return(final_df)
}



# ============================================================
# Function: compile_low_files()
# ============================================================
# Purpose:
#   Iterates over all participants to run the low SRL flagging function
#   (`flag_messages_low_srl`) and combine results into a single dataframe.
#
# Arguments:
#   outdir : Output directory containing participant CSV files
#
# Returns:
#   A combined dataframe of all low SRL flagging results.
# ============================================================
compile_low_files = function(outdir) {
  
  low_srl_flagged = NULL
  flag_list = list()
  
  for (id in unique(maps_validation_df$subjectID)) {
    cat("Processing subjectID:", id, "\n")
    
    tmp_flag = tryCatch({
      flag_messages_low_srl(output_dir = outdir, id = id)
    }, error = function(e) {
      message(sprintf("Error processing subjectID %s: %s", id, e$message))
      return(NULL)
    })
    
    if (!is.null(tmp_flag)) {
      flag_list[[as.character(id)]] = tmp_flag
    }
  }
  
  # Combine all results into one dataframe
  low_srl_flagged = do.call(rbind, flag_list)
}



# ============================================================
# Function: flag_messages_low_srl()
# ============================================================
# Purpose:
#   Flags messages as suicide-related language (SRL) based on lexicon
#   count and semantic similarity thresholds across multiple constructs.
#
# Arguments:
#   output_dir : Directory containing lexicon count and similarity CSVs
#   id         : Participant ID
#
# Returns:
#   A dataframe with suicide-related language flags per message.
# ============================================================
flag_messages_low_srl = function(output_dir, id) {
  
  # Read SRL lexicon count and similarity feature files
  count = read.csv(paste0(output_dir, id, '_srl_lexicon_counts.csv'))
  print(nrow(count))
  
  # Create flag for low SRL presence across constructs
  count = mutate(count, low_srl_allconstructs = if_any(all_of(srl_colnames), ~ .x == 1), 1, 0)
  
  similarity = read.csv(paste0(output_dir, id, '_srl_lexicon_cts_features_sentence.csv'))
  
  # Select relevant variables
  count = dplyr::select(count, subjectID, contains('tm_message'), text_clean, id_app,
                        low_srl_allconstructs, contains('suicid'))
  
  similarity = dplyr::select(similarity, subjectID, contains('tm_message'), text_clean, id_app,
                             # Max similarity columns
                             Passive.suicidal.ideation_max_s = Passive.suicidal.ideation_max,
                             Lethal.means.for.suicide_max_s = Lethal.means.for.suicide_max,
                             Active.suicidal.ideation...suicidal.planning_max_s = Active.suicidal.ideation...suicidal.planning_max,
                             Suicide.exposure_max_s = Suicide.exposure_max,
                             Other.suicidal.language_max_s = Other.suicidal.language_max,
                             # Mean similarity columns
                             Passive.suicidal.ideation_mean_s = Passive.suicidal.ideation_mean,
                             Lethal.means.for.suicide_mean_s = Lethal.means.for.suicide_mean,
                             Active.suicidal.ideation...suicidal.planning_mean_s = Active.suicidal.ideation...suicidal.planning_mean,
                             Suicide.exposure_mean_s = Suicide.exposure_mean,
                             Other.suicidal.language_mean_s = Other.suicidal.language_mean)
  
  # Merge count and similarity data
  for_validation = dplyr::left_join(count, similarity,
                                    by = c('subjectID', 'tm_message_start',
                                           'tm_message_end', 'text_clean', 'id_app'))
  
  # Identify relevant columns for flagging
  count_flag_cols = names(count)[grepl('suicid', tolower(names(count)))]
  similarity_max_cols = names(for_validation)[grepl('_max_s$', names(for_validation))]
  similarity_mean_cols = names(for_validation)[grepl('_mean_s$', names(for_validation))]
  
  # Create binary SRL flags
  for_validation = for_validation %>%
    mutate(
      low_srl_flag_count_only = if_else(
        if_any(all_of(count_flag_cols), ~ .x == 1), 1, 0
      ),
      low_srl_flag_similarity_only = if_else(
        if_any(all_of(similarity_max_cols), ~ .x > 0.7) |
          if_any(all_of(similarity_mean_cols), ~ .x > 0.5), 1, 0
      ),
      low_srl_flag = if_else(
        low_srl_flag_count_only == 1 | low_srl_flag_similarity_only == 1, 1, 0
      )
    )
  
  return(for_validation)
}
