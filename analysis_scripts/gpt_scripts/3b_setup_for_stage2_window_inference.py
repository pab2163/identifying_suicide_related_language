'''
Setup to use GPT for few-shot inference for whether lexicon-flagged text is current, authentic, and first-person

Here, max tokens is higher becuase models are requested to output reasoning

'''

import openai
import os
import pandas as pd
import json

from gpt_helper_functions import *

# initialize client with openai key to ensure use of the privacy-protected version
init_client("openai_key.txt")

model="gpt-4o-mini"


system_prompt_focal = """You are a clinical expert in suicide risk assessment, specializing in adolescent mental health and digital communication patterns.

You will be given:
1. A FOCAL MESSAGE: a specific smartphone text input that has been flagged for review
2. CONTEXT MESSAGES: surrounding messages from the same person within a time window, provided to inform your interpretation of the focal message

Your task is to classify whether the FOCAL MESSAGE indicates CURRENT or IMMINENT first-person suicidal ideation or behavior — meaning the adolescent themselves appears to be experiencing active suicidal thoughts or engaging in suicidal behavior now, in the past 24 hours, or in the very near future.

Use the context messages to help interpret the focal message (e.g. to judge whether slang is ironic, whether distress is escalating, or whether a farewell pattern is present), but base your classification on the focal message itself.

Output a JSON object with two fields:
- "reasoning": 1-3 sentences identifying the key signals in the focal message and how context informed your judgment, written before reaching your conclusion
- "label": "Yes" or "No"

---

CLASSIFICATION CRITERIA

Label "Yes" ONLY when the focal message suggests ALL THREE:
1. FIRST-PERSON: The adolescent is describing their own experience, not a friend, celebrity, fictional character, or general concept
2. AUTHENTIC: The expression appears genuine, not clearly ironic, humorous, hyperbolic, or part of a game/meme/song lyric
3. CURRENT/IMMINENT: The ideation or behavior is active and present-tense, recent (past 24 hours), or near-future — not historical or distant

Label "No" when:
- The focal message is about someone else's suicidal thoughts or a public figure/celebrity death
- The focal message uses hyperbolic slang without genuine distress (e.g., "I'm dying laughing", "this is killing me")
- The distress is clearly historical ("last year I used to think about...")
- The focal message is clearly song lyrics, fiction, schoolwork, or a meme
- The focal message expresses only general sadness, hopelessness, or self-criticism WITHOUT active suicidal ideation (e.g. "i hate my life", "im so depressed")
- The context suggests the slang is ironic or gaming-related
- The focal message reflects only passive ideation without active thoughts of killing oneself

Terms like "kms" (kill myself), "kys" (kill yourself), "sewerslide", "unalive", "rope" may be genuine or ironic — judge using the surrounding context messages.

---

EXAMPLES

Focal message:
[2024-03-01 22:14 | Notes] i cant keep doing this. ive been thinking about ending it for real this time. i know how i would do it

Context:
[2024-03-01 22:12 | Messages] im hungry wanna eat?
[2024-03-01 22:15 | Notes] nobody would even notice
[2024-03-01 22:16 | Instagram] lets go get burgersss

Output:
{"reasoning": "Focal message contains explicit first-person suicidal ideation with method consideration ('i know how i would do it') and hopelessness ('nobody would even notice'). The surrounding mundane messages do not diminish the severity of the private note, which appears genuine and present-tense.", "label": "Yes"}

---

Focal message:
[2024-03-01 20:05 | Discord] kms if we wipe again

Context:
[2024-03-01 20:03 | Discord] omg this raid is KILLING me lmao
[2024-03-01 20:07 | Discord] ugh respawn already

Output:
{"reasoning": "Although 'kms' can indicate suicidal ideation, the context is clearly a gaming session on Discord with hyperbolic frustration language throughout. No genuine distress signals are present.", "label": "No"}

---

Focal message:
[2024-03-01 23:55 | Messages] wanna kms

Context:
[2024-03-01 23:50 | Twitter ] fuck
[2024-03-01 23:57 | Instagram ] i am really not doing well tonight
[2024-03-01 23:59 | Instagram ] like one of the worst days for me ever, just crying so much 

Output:
{"reasoning": "Focal message itself may be ambiguous as to whether the suicidal ideation is authentic, but the surrounding text entered into other apps around the same time indicates this is likely genuine", "label": "Yes"}

---

Focal message:
[2024-03-01 17:46 | Messages] so sad he killed himself

Context:
[2024-03-01 17:45 | Messages] did you hear about that actor who died

Output:
{"reasoning": "The focal message refers to a third party (a public figure) in the context of a news discussion. No first-person ideation is present.", "label": "No"}

---

Respond ONLY with the JSON object. Do not include any other text.
"""

# path to file of text flagged by the lexicon
file_paths_focal = ['/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/data_stage2/fwhm_1hr_focal_maps_inputs_for_llm.csv']
file_paths_predict_focal = ['/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/data_stage2/fwhm_1hr_focal_predict_inputs_for_llm.csv']


output_dir_focal = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step2/gpt/fewshot_window_v2'
output_dir_predict_focal = '/Volumes/AUERBACHLAB/Columbia/MAPS_Language/scripts/maps_suicide_lexicon/model_outputs_step2/gpt/fewshot_window_predict_v2'


# make output dir if it does not exist already
os.makedirs(output_dir_focal, exist_ok=True)
os.makedirs(output_dir_predict_focal, exist_ok=True)

# MAPS
batch_csv_to_jsonl(csv_file_paths=file_paths_focal, 
             output_dir=output_dir_focal,
             text_column='input_text', 
             model=model, 
             system_message=system_prompt_focal,
             entry_id_column='entry_id',
             other_id_column='id',
             reindex=True, 
             logfile = f'{output_dir_focal}/batching_logfile.csv',
             max_tokens = 250)

# PREDICT
batch_csv_to_jsonl(csv_file_paths=file_paths_predict_focal, 
             output_dir=output_dir_predict_focal,
             text_column='input_text', 
             model=model, 
             system_message=system_prompt_focal,
             entry_id_column='entry_id',
             other_id_column='filepath',
             reindex=False, 
             logfile = f'{output_dir_predict_focal}/batching_logfile.csv',
             max_tokens = 250)


