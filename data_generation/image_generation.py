import torch
from datetime import datetime
import os
from huggingface_hub import login
from diffusers import Flux2KleinPipeline
import transformers
import shutil

login(token="YOUR-HF-TOKEN") # Flux2 requires the acceptance of terms and conditions by the HF account connected to this token

PROMPTS_DIR = "./prompts"   # path to the folder with the prompts' txt files
OUTPUT_DIR = "./images"     # path to the folder in which to save the images

LOG_FILE = "generation_log.txt"

# Importing prompts from txt files
with open(f'{PROMPTS_DIR}/amb.txt', 'r') as f:
    amb = list(line.strip('\n') for line in f)
amb = amb[::3] # take only one line every 3 (the ambiguous prompt, the other 2 lines are the 2 meaning of that sentence)

with open(f'{PROMPTS_DIR}/ldd.txt', 'r') as f:
    ldd = list(line.strip('\n') for line in f)

with open(f'{PROMPTS_DIR}/wo.txt', 'r') as f:
    wo = list(line.strip('\n') for line in f)

PROMPTS = {'WordOrder':wo, 'Ambiguity': amb, 'LDD': ldd}
METRICS = ['WordOrder', 'Ambiguity', 'LDD']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def log_generation(prompt, metric, filepath):
    '''Writes a log of prompts for which an image has been generated.'''
    with open(LOG_FILE, "a") as log:
        log.write(f"{datetime.now()} | {metric} | {prompt} | {filepath}\n")

def load_log():
    '''Reads the log of already generated prompts.'''
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE) as log:
        return set(line.split(" | ")[2] for line in log)  # return already generated prompts


# Loading Flux2
device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
pipe = pipe.to(device)


# Generating images
already_generated = load_log()

for metric in METRICS:
    for i, prompt in enumerate(PROMPTS[metric]):
        if prompt in already_generated:
            continue

        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator().manual_seed(0)
        ).images[0]

        filename = f"{metric}_{i}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)

        log_generation(prompt, metric, filepath)

# Creating zip archive of the images
shutil.make_archive("Images", 'zip', OUTPUT_DIR)