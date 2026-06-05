QUESTIONS_DIR = "./questions" # path to the folder with the csvs with the questions
IMAGES_DIR    = "./images"    # path to the folder with the images
OUTPUT_DIR    = "./results"   # path to the folder in which to save the results

import pandas as pd
import os
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

# Import the questions for the VLM
word_order  = pd.read_csv(f'{QUESTIONS_DIR}/word_order_df.csv')
ldd         = pd.read_csv(f'{QUESTIONS_DIR}/ldd_df.csv')
ambiguity   = pd.read_csv(f'{QUESTIONS_DIR}/ambiguity_df.csv')

wo_img  = []
ldd_img = []
amb_img = []

for i in range(len(word_order)):
  wo_img.append(f'WordOrder_{i}.png')

for i in range(len(ldd)):
  ldd_img.append(f'LDD_{i}.png')

for i in range(len(ambiguity)):
  amb_img.append(f'Ambiguity_{i}.png')

word_order['img_name'] = wo_img
ldd['img_name'] = ldd_img
ambiguity['img_name'] = amb_img

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Loading VLM (Qwen3)
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


# Support functions

def answer_question(question:str, image_path:str):
  '''Give Qwen3 a question and the path to an image and record the answer'''
  messages = [
      {
          "role": "user",
          "content": [{
                  "type": "image",
                  "image": image_path
              },

              {"type": "text", "text": question},
          ],
      }
  ]

  # Preparation for inference
  inputs = processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_dict=True,
      return_tensors="pt"
  )
  inputs = inputs.to(model.device)

  # Inference: Generation of the output
  generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
  )
  return(output_text[0])

def test_wordorder(dataframe:pd.DataFrame, image_folder:str):
  '''Ask questions from the (word order) dataframe to Qwen and returns a dataframe with the answers'''
  df = dataframe.copy()
  for i in range(len(df)):
    image_path = f"{image_folder}/{df['img_name'].iloc[i]}"
    if type(image_path)!=str:
      print(f'Error at line {i}: image path is not a string.')
      continue
    for q,a in zip(['q1','q2','q3','cq3'], ['a1', 'a2', 'a3', 'ca3']):
      question = df[q].iloc[i]
      if type(question)!=str:
        print(f'Error at line {i}: question {q} is not a string.')
        continue
      df.at[i, a] = answer_question(question, image_path)
    if i%10==0:
      print(f'Done word order prompt {i}')
  return(df)

def test_LDD(dataframe:pd.DataFrame, image_folder:str):
  '''Ask questions from the (long distance dependencies) dataframe to Qwen and returns a dataframe with the answers'''
  df = dataframe.copy()
  for i in range(len(df)):
    image_path = f"{image_folder}/{df['img_name'].iloc[i]}"
    if type(image_path)!=str:
      print(f'Error at line {i}: image path is not a string.')
      continue
    for q,a in zip(['q1','q2','q3','q4','q5','q6'], ['a1','a2','a3','a4','a5','a6']):
      question = df[q].iloc[i]
      if type(question)!=str:
        continue
      df.at[i, a] = answer_question(question, image_path)
    if i%10==0:
      print(f'Done LDD prompt {i}')
  return(df)

def test_ambiguity(dataframe:pd.DataFrame, image_folder:str):
  '''Ask questions from the (ambiguity) dataframe to Qwen and returns a dataframe with the answers'''
  df = dataframe.copy()
  for i in range(len(df)):
    image_path = f"{image_folder}/{df['img_name'].iloc[i]}"
    if type(image_path)!=str:
      print(f'Error at line {i}: image path is not a string.')
      continue
    for q,a in zip(['q1','q2','q3','q4','cq4'], ['a1','a2','a3','a4','ca4']):
      question = df[q].iloc[i]
      if type(question)!=str:
        print(f'Error at line {i}: question {q} is not a string.')
        continue
      df.at[i, a] = answer_question(question, image_path)
    if i%10==0:
      print(f'Done ambiguity prompt {i}')

  return(df)

def test_all(word_order_df:pd.DataFrame, ldd_df:pd.DataFrame, ambiguity_df:pd.DataFrame, image_folder=IMAGES_DIR):
  '''Runs the test functions for the 3 subexperiments'''
  word_order_res = test_wordorder(word_order_df, image_folder)
  word_order_res.to_csv(f'{OUTPUT_DIR}/res_wo.csv', index=False)
  ldd_res = test_LDD(ldd_df, image_folder)
  ldd_res.to_csv(f'{OUTPUT_DIR}/res_ldd.csv', index=False)
  ambiguity_res = test_ambiguity(ambiguity_df, image_folder)
  ambiguity_res.to_csv(f'{OUTPUT_DIR}/res_amb.csv', index=False)
  return(word_order_res, ldd_res, ambiguity_res)

# Running the tests
res_wo, res_ldd, res_amb = test_all(word_order, ldd, ambiguity)