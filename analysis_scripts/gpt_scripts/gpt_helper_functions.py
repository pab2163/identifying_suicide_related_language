"""
batch_utils.py
==============
Utilities for running OpenAI batch classification jobs against CSV data.

Typical workflow
----------------
1. init_client()                        — load API key, create OpenAI client
2. batch_csv_to_jsonl(...)              — convert one or more CSVs to JSONL batch files
3. launch_multiple_batch_requests(...)  — upload and submit each JSONL as a batch job
4. retrieve_all_batches(...)            — poll and download completed results
5. process_all_inputs(...)              — parse JSONL responses and join back to source CSVs

Logfile (default: batch_job_tracking.csv)
-----------------------------------------
A CSV that tracks every JSONL file from creation through retrieval. Columns:
  input_jsonl, source_csv, jsonl_created_timestamp, batch_id, file_id,
  description, launch_timestamp, retrieval_status, retrieval_timestamp,
  output_path, retrieval_error
"""

import openai
import os
import pandas as pd
import json
import math
import re

client = None


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def init_client(api_key_file="openai_key.txt"):
    """
    Initialise the global OpenAI client from a plaintext key file.

    Parameters
    ----------
    api_key_file : str
        Path to a file whose only content is the OpenAI API key.

    Returns
    -------
    openai.OpenAI
        The initialised client (also stored in the module-level ``client``).
    """
    global client
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()
    client = openai.OpenAI(api_key=api_key, timeout=60)
    print(f"✓ OpenAI client initialized")
    return client


def _check_client():
    """Raise RuntimeError if init_client() has not been called yet."""
    if client is None:
        raise RuntimeError("OpenAI client not initialized — call init_client('openai_key.txt') first.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_nan(x):
    """Return True if *x* is a float NaN (distinguishes from None / empty string)."""
    return isinstance(x, float) and math.isnan(x)


def robust_read(input_csv):
    """
    Read a CSV, gracefully handling UTF-8 encoding errors.

    Rows containing the replacement character ``â`` (a common UTF-8 mojibake
    artefact) are cleaned by stripping the offending character.

    Parameters
    ----------
    input_csv : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        print("Encoding error detected, replacing problematic characters...")
        df = pd.read_csv(input_csv, encoding='utf-8', encoding_errors='replace')
        problematic_rows = df.index[df.apply(lambda x: x.astype(str).str.contains('â').any(), axis=1)]
        for row in problematic_rows:
            df.loc[row] = df.loc[row].apply(lambda x: str(x).replace('â', '') if isinstance(x, str) else x)
        print(f"Fixed {len(problematic_rows)} problematic rows")
    return df


# ---------------------------------------------------------------------------
# JSONL creation
# ---------------------------------------------------------------------------

def csv_to_jsonl(input_csv: str, output_jsonl: str, text_column='corrected_message',
                 model=None, system_message=None, entry_id_column='entry_id',
                 other_id_column=None, max_tokens=20):
    """
    Convert a CSV file to one or more OpenAI batch-API JSONL files.

    Each row becomes one chat-completion request. Files larger than 50 000 rows
    are split into numbered parts (``_part1.jsonl``, ``_part2.jsonl``, …) to
    stay within OpenAI's per-file limits.

    Parameters
    ----------
    input_csv : str
        Path to the source CSV.
    output_jsonl : str
        Destination path for the JSONL file (base name used when splitting).
    text_column : str
        Column containing the text to classify.
    model : str
        OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
    system_message : str
        System prompt sent with every request. Must not be None.
    entry_id_column : str
        Column used as the primary request identifier.
    other_id_column : str or None
        Optional secondary ID column (appended to ``custom_id``).
    max_tokens : int
        ``max_tokens`` passed to the completion endpoint.
    """
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        print("Encoding error replaced")
        df = pd.read_csv(input_csv, encoding='utf-8', encoding_errors='replace')
        # FIX: consistent mojibake check — use 'â' throughout
        problematic_rows = df.index[df.apply(lambda x: x.astype(str).str.contains('â').any(), axis=1)]
        for row in problematic_rows:
            df.loc[row] = df.loc[row].apply(
                lambda x: str(x).replace('â', '') if isinstance(x, str) else x
            )

    cols = [entry_id_column, text_column] + ([other_id_column] if other_id_column else [])
    df = df.dropna(subset=cols)

    if len(df) > 50000:
        print(f'Large file: {len(df)} rows — will be split into parts.')

    def write_chunk(chunk_df, output_file, max_tokens):
        with open(output_file, 'w') as f:
            for idx, row in chunk_df.iterrows():
                text_value = row[text_column]
                entry_id = row[entry_id_column]

                if is_nan(text_value):
                    print(f"Skipping row {idx}: text is NaN")
                    continue

                if is_nan(entry_id):
                    print(f"Skipping row {idx}: entry_id is NaN")
                    continue

                if system_message is None:
                    raise ValueError("system_message is None → will create invalid batch")

                if other_id_column:
                    id_string = f'request_id-{row[entry_id_column]}_secondaryid-{row[other_id_column]}'
                else:
                    id_string = f'request_id-{row[entry_id_column]}'

                batch_request = {
                    "custom_id": id_string,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": row[text_column]}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0
                    }
                }
                json.dump(batch_request, f)
                f.write('\n')

    if len(df) > 50000:
        num_parts = (len(df) + 49999) // 50000
        rows_per_part = len(df) // num_parts
        for part in range(num_parts):
            start_idx = part * rows_per_part
            end_idx = len(df) if part == num_parts - 1 else (part + 1) * rows_per_part
            output_file = output_jsonl.replace('.jsonl', f'_part{part+1}.jsonl')
            write_chunk(df.iloc[start_idx:end_idx], output_file, max_tokens=max_tokens)
    else:
        write_chunk(df, output_jsonl, max_tokens=max_tokens)


def test_jsonl_system_prompt_length(jsonl_file):
    """
    Print the character length of the system prompt in each request in a JSONL file.

    Useful for checking that prompts are within API limits before submitting.

    Parameters
    ----------
    jsonl_file : str
        Path to the JSONL file to inspect.
    """
    with open(jsonl_file, "r") as f:
        for i, line in enumerate(f):
            request = json.loads(line)
            system_msg = next(m["content"] for m in request["body"]["messages"] if m["role"] == "system")
            print(f"Request {i}: {len(system_msg)} chars")


# ---------------------------------------------------------------------------
# Batch CSV → JSONL (multi-file)
# ---------------------------------------------------------------------------

def batch_csv_to_jsonl(
    csv_file_paths: list,
    output_dir: str,
    text_column='corrected_message',
    model=None,
    system_message=None,
    entry_id_column='entry_id',
    other_id_column=None,
    reindex=True,
    logfile='batch_job_tracking.csv',
    max_tokens=20
):
    """
    Convert multiple CSV files to JSONL batch files and seed the tracking logfile.

    For each CSV the function:
    - Optionally rewrites ``entry_id_column`` as a clean 0-based integer index
      (``reindex=True``, the default).
    - Saves the annotated CSV to ``<output_dir>/input_csv/``.
    - Calls :func:`csv_to_jsonl` to produce one or more JSONL files in
      ``<output_dir>/input_jsonl/``.
    - Appends a row per JSONL file to the tracking logfile.

    Parameters
    ----------
    csv_file_paths : list of str
        Paths to the input CSV files.
    output_dir : str
        Root directory for all outputs.
    text_column : str
        Column containing the text to classify.
    model : str
        OpenAI model identifier.
    system_message : str
        System prompt for every request.
    entry_id_column : str
        Name of the primary ID column.
    other_id_column : str or None
        Optional secondary ID column.
    reindex : bool
        If True (default), overwrite ``entry_id_column`` with a fresh 0-based index.
    logfile : str
        Path to the CSV tracking logfile.
    max_tokens : int
        ``max_tokens`` forwarded to the completion endpoint.

    Returns
    -------
    dict
        ``{"success": [...], "failed": [...]}`` with per-file details.
    """
    csv_out_dir = os.path.join(output_dir, "input_csv")
    jsonl_out_dir = os.path.join(output_dir, "input_jsonl")
    os.makedirs(csv_out_dir, exist_ok=True)
    os.makedirs(jsonl_out_dir, exist_ok=True)

    results = {"success": [], "failed": []}

    for csv_path in csv_file_paths:
        if not os.path.exists(csv_path):
            print(f"[SKIP] File not found: {csv_path}")
            results["failed"].append({"file": csv_path, "reason": "File not found"})
            continue

        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[PROCESSING] {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"  Encoding error detected, replacing problematic characters.")
            df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')

        problematic_rows = df.index[
            df.apply(lambda x: x.astype(str).str.contains('â').any(), axis=1)
        ]
        for row in problematic_rows:
            df.loc[row] = df.loc[row].apply(
                lambda x: str(x).replace('â', '') if isinstance(x, str) else x
            )

        if reindex:
            if entry_id_column in df.columns:
                print(f"  Warning: Column '{entry_id_column}' already exists — overwriting.")
            df.insert(0, entry_id_column, range(len(df)))
            annotated_csv_path = os.path.join(csv_out_dir, f"{csv_name}_with_ids.csv")
            df.to_csv(annotated_csv_path, index=False, encoding='utf-8')
            print(f"  ✓ Saved annotated CSV: {annotated_csv_path}")
            source_csv_path = annotated_csv_path
        else:
            source_csv_path = csv_path

        output_jsonl = os.path.join(jsonl_out_dir, f"{csv_name}.jsonl")

        try:
            csv_to_jsonl(
                input_csv=source_csv_path,
                output_jsonl=output_jsonl,
                text_column=text_column,
                model=model,
                system_message=system_message,
                entry_id_column=entry_id_column,
                other_id_column=other_id_column,
                max_tokens=max_tokens
            )
            generated = [
                os.path.join(jsonl_out_dir, f) for f in os.listdir(jsonl_out_dir)
                if f.startswith(csv_name) and f.endswith('.jsonl')
            ]
            print(f"  ✓ Generated JSONL: {generated}")
            results["success"].append({
                "file": csv_path,
                "annotated_csv": source_csv_path if reindex else None,
                "outputs": generated
            })

            new_rows = pd.DataFrame([{
                "input_jsonl": jsonl_path,
                "source_csv": source_csv_path,
                "jsonl_created_timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "batch_id": None,
                "file_id": None,
                "description": None,
                "launch_timestamp": None,
                "retrieval_status": None,
                "retrieval_timestamp": None,
                "output_path": None,
                "retrieval_error": None,
            } for jsonl_path in generated])

            if os.path.exists(logfile):
                new_rows.to_csv(logfile, mode='a', header=False, index=False)
            else:
                new_rows.to_csv(logfile, mode='w', header=True, index=False)

        except Exception as e:
            print(f"  ✗ JSONL generation failed: {e}")
            results["failed"].append({"file": csv_path, "reason": str(e)})

    print(f"\n{'='*50}")
    print(f"Done. {len(results['success'])} succeeded, {len(results['failed'])} failed.")
    if results["failed"]:
        print("Failed files:")
        for f in results["failed"]:
            print(f"  - {f['file']}: {f['reason']}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    return results


# ---------------------------------------------------------------------------
# Launching batch jobs
# ---------------------------------------------------------------------------

def launch_batch_request_from_jsonl(jsonl_filepath, description='Suicide Language Classification',
                                    logfile='batch_job_tracking.csv'):
    """
    Upload a JSONL file to the Files API and submit it as a batch job.

    Updates the logfile row matching ``jsonl_filepath`` with the resulting
    ``batch_id``, ``file_id``, ``description``, and ``launch_timestamp``.
    If no matching row exists (e.g. the file was created outside
    :func:`batch_csv_to_jsonl`), a new row is appended.

    Parameters
    ----------
    jsonl_filepath : str
        Path to the JSONL file to submit.
    description : str
        Human-readable label stored in the batch metadata and logfile.
    logfile : str
        Path to the CSV tracking logfile.
    """
    _check_client()

    batch_input_file = client.files.create(
        file=open(jsonl_filepath, "rb"),
        purpose='batch'
    )
    batch_input_file_id = batch_input_file.id

    batch_response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window='24h',
        metadata={"description": description}
    )

    batch_id = batch_response.id
    print(f'Launched batch id {batch_id}!')

    log_df = pd.read_csv(logfile) if os.path.exists(logfile) else pd.DataFrame()

    for col in ['batch_id', 'file_id', 'description', 'launch_timestamp']:
        if col in log_df.columns:
            log_df[col] = log_df[col].astype(str).where(log_df[col].notna(), other=None)

    if len(log_df) and 'input_jsonl' in log_df.columns and jsonl_filepath in log_df['input_jsonl'].values:
        log_df.loc[log_df['input_jsonl'] == jsonl_filepath, 'batch_id'] = batch_id
        log_df.loc[log_df['input_jsonl'] == jsonl_filepath, 'file_id'] = batch_input_file_id
        log_df.loc[log_df['input_jsonl'] == jsonl_filepath, 'description'] = description
        log_df.loc[log_df['input_jsonl'] == jsonl_filepath, 'launch_timestamp'] = \
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        new_row = pd.DataFrame([{
            "input_jsonl": jsonl_filepath,
            "source_csv": None,
            "jsonl_created_timestamp": None,
            "batch_id": batch_id,
            "file_id": batch_input_file_id,
            "description": description,
            "launch_timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "retrieval_status": None,
            "retrieval_timestamp": None,
            "output_path": None,
            "retrieval_error": None,
        }])
        log_df = pd.concat([log_df, new_row], ignore_index=True)

    log_df.to_csv(logfile, index=False)


def launch_multiple_batch_requests(jsonl_filepaths: list,
                                   description='Suicide Language Classification',
                                   logfile='batch_job_tracking.csv'):
    """
    Submit multiple JSONL files as separate batch jobs.

    Calls :func:`launch_batch_request_from_jsonl` for each file in order.

    Parameters
    ----------
    jsonl_filepaths : list of str
        Paths to the JSONL files to submit.
    description : str
        Human-readable label applied to every batch.
    logfile : str
        Path to the CSV tracking logfile.
    """
    for jsonl_filepath in jsonl_filepaths:
        launch_batch_request_from_jsonl(jsonl_filepath, description=description, logfile=logfile)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_batch(batch_id, output_dir):
    """
    Download the output file for a single completed batch job.

    Parameters
    ----------
    batch_id : str
        The OpenAI batch ID to retrieve.
    output_dir : str
        Directory where the output JSONL will be written.

    Returns
    -------
    str or None
        Path to the saved JSONL file, or None if no output file is available
        (e.g. the batch has not finished yet).
    """
    _check_client()

    os.makedirs(output_dir, exist_ok=True)

    batch_status = client.batches.retrieve(batch_id)
    output_file_id = batch_status.output_file_id

    if not output_file_id:
        print(f"  [SKIP] Batch {batch_id} has no output file — status: {batch_status.status}")
        return None

    output_file = client.files.content(output_file_id).content
    output_jsonl_path = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")
    with open(output_jsonl_path, "w") as f:
        f.write(output_file.decode('utf-8'))
    print(f"  ✓ Saved JSONL: {output_jsonl_path}")

    return output_jsonl_path


def retrieve_all_batches(logfile, output_dir):
    """
    Attempt retrieval for every batch in the logfile not yet marked ``success``.

    For each batch the function:
    - Prints status and request counts.
    - Downloads the error file if one exists.
    - Downloads the output JSONL on success.
    - Updates the logfile row for that batch only.

    Parameters
    ----------
    logfile : str
        Path to the CSV tracking logfile.
    output_dir : str
        Directory where output and error JSONL files will be written.

    Returns
    -------
    dict
        ``{"success": [...], "failed": [...]}`` with per-batch details.
    """
    _check_client()
    if not os.path.exists(logfile):
        raise FileNotFoundError(f"Logfile not found: {logfile}")

    log_df = pd.read_csv(logfile)

    for col in ['retrieval_status', 'output_path', 'retrieval_error']:
        log_df[col] = log_df[col].astype(str).where(log_df[col].notna(), other=None)

    if 'batch_id' not in log_df.columns:
        raise ValueError("Logfile must contain a 'batch_id' column")

    retrievable = log_df[
        log_df['batch_id'].notna() &
        (log_df['retrieval_status'] != 'success')
    ]
    print(f"Found {len(retrievable)} batch(es) pending retrieval.")

    results = {"success": [], "failed": []}

    for _, row in retrievable.iterrows():
        batch_id = row['batch_id']
        print(f"\n[RETRIEVING] {batch_id}")
        try:
            batch_status = client.batches.retrieve(batch_id)

            print(f"  Status: {batch_status.status}")
            print(f"  Requests — total: {batch_status.request_counts.total}, "
                  f"completed: {batch_status.request_counts.completed}, "
                  f"failed: {batch_status.request_counts.failed}")

            if batch_status.error_file_id:
                error_content = client.files.content(batch_status.error_file_id).content
                error_file_path = os.path.join(output_dir, f"batch_errors_{batch_id}.jsonl")
                os.makedirs(output_dir, exist_ok=True)
                with open(error_file_path, "w") as f:
                    f.write(error_content.decode('utf-8'))
                print(f"  ⚠ Error file saved: {error_file_path}")
                log_df.loc[log_df['batch_id'] == batch_id, 'error_file_path'] = error_file_path

            path = retrieve_batch(batch_id=batch_id, output_dir=output_dir)
            if path:
                results["success"].append({"batch_id": batch_id, "output": path})
                log_df.loc[log_df['batch_id'] == batch_id, 'retrieval_status'] = 'success'
                log_df.loc[log_df['batch_id'] == batch_id, 'output_path'] = path
            else:
                results["failed"].append({"batch_id": batch_id, "reason": f"No output file — status: {batch_status.status}"})
                log_df.loc[log_df['batch_id'] == batch_id, 'retrieval_status'] = batch_status.status

            log_df.loc[log_df['batch_id'] == batch_id, 'retrieval_timestamp'] = \
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results["failed"].append({"batch_id": batch_id, "reason": str(e)})
            log_df.loc[log_df['batch_id'] == batch_id, 'retrieval_status'] = 'failed'
            log_df.loc[log_df['batch_id'] == batch_id, 'retrieval_error'] = str(e)
            log_df.loc[log_df['batch_id'] == batch_id, 'retrieval_timestamp'] = \
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    log_df.to_csv(logfile, index=False)
    print(f"  ✓ Updated logfile: {logfile}")

    print(f"\n{'='*50}")
    print(f"Done. {len(results['success'])} succeeded, {len(results['failed'])} failed.")
    if results["failed"]:
        print("Failed batches:")
        for f in results["failed"]:
            print(f"  - {f['batch_id']}: {f['reason']}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    return results


# ---------------------------------------------------------------------------
# Parsing and joining
# ---------------------------------------------------------------------------

def parse_jsonl_output(jsonl_path):
    """
    Parse a batch output JSONL file into a DataFrame.

    Each line is expected to be a JSON object with a ``custom_id`` and a
    ``response`` containing a chat completion. The response content is parsed
    as either:

    - A JSON object with ``label`` and optionally ``reasoning`` keys, or
    - Plain text, from which a ``Yes``/``No`` label is extracted with regex.

    Labels are normalised to ``"Yes"`` or ``"No"``; anything else becomes None.
    Lines with errors or missing ``response`` keys are skipped with a warning.

    Parameters
    ----------
    jsonl_path : str
        Path to the batch output JSONL file.

    Returns
    -------
    pd.DataFrame
        Columns: ``custom_id``, ``label``, ``reasoning``.
    """
    def parse_response(response_str):
        label = None
        reasoning = None
        try:
            parsed = json.loads(response_str)
            if isinstance(parsed, dict):
                label = parsed.get("label")
                reasoning = parsed.get("reasoning")
            else:
                raise ValueError(f"Unexpected JSON type: {type(parsed)}")
        except (json.JSONDecodeError, ValueError):
            label_match = re.search(r'\b(Yes|No)\b', response_str, re.IGNORECASE)
            label = label_match.group(1).capitalize() if label_match else None
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response_str)
            reasoning = reasoning_match.group(1) if reasoning_match else None

        if label is not None:
            label = label.capitalize()
            if label not in ("Yes", "No"):
                label = None

        return label, reasoning

    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)

            if "error" in entry and entry["error"]:
                print(f"  ⚠ Skipping {entry.get('custom_id')}: {entry['error']}")
                continue

            if "response" not in entry:
                print(f"  ⚠ Skipping malformed entry: {entry.get('custom_id')} — keys: {list(entry.keys())}")
                continue

            label, reasoning = parse_response(
                entry["response"]["body"]["choices"][0]["message"]["content"]
            )
            rows.append({
                "custom_id": entry["custom_id"],
                "label": label,
                "reasoning": reasoning,
            })

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} rows from {jsonl_path}")
    return df


def join_jsonl_to_input_csv(jsonl_df, input_csv, output_csv,
                             entry_id_column='entry_id', other_id_column=None):
    """
    Left-join parsed batch responses back to the source CSV.

    If ``output_csv`` already exists (e.g. from a previous partial join), it is
    used as the base and new responses fill in only null cells — existing values
    are preserved.

    ``custom_id`` values are parsed to recover ``entry_id_column`` (and
    optionally ``other_id_column``) before merging. Both sides of the merge are
    cast to ``str`` to avoid int64/object type mismatches.

    Parameters
    ----------
    jsonl_df : pd.DataFrame
        Output of :func:`parse_jsonl_output`.
    input_csv : str
        Path to the original (annotated) source CSV.
    output_csv : str
        Path where the merged result will be written.
    entry_id_column : str
        Primary ID column name.
    other_id_column : str or None
        Optional secondary ID column name.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame (also written to ``output_csv``).
    """
    def parse_ids(custom_id):
        if '_secondaryid-' in custom_id:
            match = re.match(r'request_id-(\d+)_secondaryid-(.+)', custom_id)
            return (match.group(1), match.group(2)) if match else (None, None)
        else:
            match = re.match(r'request_id-(\d+)', custom_id)
            return (match.group(1), None) if match else (None, None)

    jsonl_df[[entry_id_column, '_secondary_parsed']] = \
        jsonl_df['custom_id'].apply(lambda x: pd.Series(parse_ids(x)))

    if other_id_column:
        jsonl_df = jsonl_df.rename(columns={'_secondary_parsed': other_id_column})
        merge_keys = [entry_id_column, other_id_column]
    else:
        jsonl_df = jsonl_df.drop(columns=['_secondary_parsed'])
        merge_keys = [entry_id_column]

    response_cols = [c for c in ['label', 'reasoning'] if c in jsonl_df.columns]
    jsonl_df = jsonl_df[merge_keys + response_cols]

    if os.path.exists(output_csv):
        print(f"  Output CSV exists — merging incrementally: {output_csv}")
        base_df = robust_read(output_csv)
    else:
        base_df = robust_read(input_csv)

    for key in merge_keys:
        jsonl_df[key] = jsonl_df[key].astype(str)
        base_df[key] = base_df[key].astype(str)

    existing_response_cols = [c for c in response_cols if c in base_df.columns]
    if existing_response_cols:
        merged_df = pd.merge(base_df, jsonl_df, on=merge_keys, how='left', suffixes=('_prior', '_new'))
        for col in existing_response_cols:
            merged_df[col] = merged_df[f'{col}_prior'].combine_first(merged_df[f'{col}_new'])
            merged_df = merged_df.drop(columns=[f'{col}_prior', f'{col}_new'])
    else:
        merged_df = pd.merge(base_df, jsonl_df, on=merge_keys, how='left')

    merged_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved: {output_csv}")
    return merged_df


# ---------------------------------------------------------------------------
# End-to-end processing
# ---------------------------------------------------------------------------

def process_all_inputs(output_dir, entry_id_column='entry_id', other_id_column=None,
                       logfile='batch_job_tracking.csv', expected_column='label'):
    """
    Parse and join all successfully retrieved batches to their source CSVs.

    Groups retrieved batches by ``source_csv``, parses each output JSONL with
    :func:`parse_jsonl_output`, and joins results back via
    :func:`join_jsonl_to_input_csv`. After processing each source CSV a missing-
    data report is printed showing how many rows lack a response.

    Batches not yet marked ``retrieval_status == 'success'`` in the logfile are
    skipped with a warning.

    Parameters
    ----------
    output_dir : str
        Directory where joined output CSVs will be written.
    entry_id_column : str
        Primary ID column used for joining.
    other_id_column : str or None
        Optional secondary ID column used for joining.
    logfile : str
        Path to the CSV tracking logfile.
    expected_column : str
        Column name to use for the missing-data report. Defaults to ``'label'``
        (the column produced by :func:`parse_jsonl_output`).

        .. note::
           The previous default was ``'response'``, which never exists in the
           output — this has been corrected to ``'label'``.

    Returns
    -------
    dict
        Mapping of ``source_csv`` path → ``output_csv`` path for each processed
        input file.
    """
    log_df = pd.read_csv(logfile)

    ready = log_df[log_df['retrieval_status'] == 'success'].copy()
    unready = log_df[log_df['retrieval_status'] != 'success']
    if len(unready):
        print(f"  ⚠ Skipping {len(unready)} batch(es) not yet retrieved: {unready['batch_id'].tolist()}")

    if ready.empty:
        print("No successfully retrieved batches to process.")
        return {}

    input_csv_to_jsonls = (
        ready.groupby('source_csv')['output_path']
        .apply(list)
        .to_dict()
    )

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for input_csv, jsonl_filepaths in input_csv_to_jsonls.items():
        print(f"\n{'='*50}")
        print(f"[INPUT CSV] {input_csv} — {len(jsonl_filepaths)} JSONL file(s)")

        input_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(output_dir, f"{input_name}_with_responses.csv")

        for jsonl_path in jsonl_filepaths:
            print(f"\n  [PROCESSING] {jsonl_path}")
            try:
                jsonl_df = parse_jsonl_output(jsonl_path)
                join_jsonl_to_input_csv(
                    jsonl_df=jsonl_df,
                    input_csv=input_csv,
                    output_csv=output_csv,
                    entry_id_column=entry_id_column,
                    other_id_column=other_id_column
                )
            except Exception as e:
                print(f"  ✗ Failed: {e}")

        if os.path.exists(output_csv):
            final_df = pd.read_csv(output_csv)
            n_missing = final_df[expected_column].isna().sum()
            total = len(final_df)
            print(f"\n  Missing response report for {output_csv}:")
            print(f"  {n_missing}/{total} rows missing responses ({100 * n_missing / total:.1f}%)")
        else:
            print(f"\n  ⚠ No output CSV found for {input_csv} — all JSONL joins may have failed.")

        results[input_csv] = output_csv

    return results