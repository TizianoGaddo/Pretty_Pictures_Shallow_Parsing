import pandas as pd
import re
import os

PROMPTS_DIR = "./prompts"   # path to the folder with the prompts' txt files
OUTPUT_DIR  = "./questions" # path to the folder in which to save the questions csv files

N_PROMPTS = 80

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Importing prompts from txt files
with open(f'{PROMPTS_DIR}/amb.txt', 'r') as f:
    amb_all = list(line.strip('\n') for line in f)
amb = amb_all[::3]

with open(f'{PROMPTS_DIR}/ldd.txt', 'r') as f:
    ldd = list(line.strip('\n') for line in f)

with open(f'{PROMPTS_DIR}/wo.txt', 'r') as f:
    wo = list(line.strip('\n') for line in f)

PROMPTS = {'WordOrder': wo, 'Ambiguity': amb, 'LDD': ldd}
METRICS = ['WordOrder', 'Ambiguity', 'LDD']

# WORD ORDER questions
word_order_df = pd.DataFrame(columns=['prompt', 'object_1', 'object_2', 'relation', 'q1', 'q2', 'q3', 'cq3'])

for prompt in PROMPTS['WordOrder']:
    matches = re.search(r'(A |An )([\w\s]+) is (on top of|under|next to|to the left of|to the right of) (a |an )([\w\s]+)', prompt)
    art1 = matches.group(1)
    object1 = matches.group(2)
    relation = matches.group(3)
    art2 = matches.group(4)
    object2 = matches.group(5)
    q1 = 'Is there ' + art1.lower() + object1 + ' in the image? Please answer only with a yes or no.'
    q2 = 'Is there ' + art2 + object2 + ' in the image? Please answer only with a yes or no.'
    q3 = 'Is the ' + object1 + ' ' + relation + ' the ' + object2 + '? Please answer only with a yes or no.'
    cq3 = 'Is the ' + object2 + ' ' + relation + ' the ' + object1 + '? Please answer only with a yes or no.' # control question (order of the objects is inverted)
    word_order_df.loc[len(word_order_df)] = [prompt, object1, object2, relation, q1, q2, q3, q4]

prompt_index = [i for i in range(1, N_PROMPTS+1) for _ in range(2)]
word_order_df['prompt_index'] = prompt_index
word_order_df.to_csv(f'{OUTPUT_DIR}/word_order_df.csv', index=False)

# LDD questions
LDD_df = pd.DataFrame(columns=['prompt', 'main_subj', 'main_verb', 'sub_subj', 'sub_verb', 'attribute_1', 'attribute_2', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'])
for prompt in PROMPTS['LDD']:
  matches = re.search(r'The (.*?) that the (.*?)(?: with (.*?))?(?: and (.*?))? is (.*?) is (.*?)\.', prompt)
  main_subj = matches.group(1)
  main_verb = matches.group(6)
  sub_subj = matches.group(2)
  sub_verb = matches.group(5)
  attribute1 = matches.group(3)
  attribute2 = matches.group(4)
  q1 = 'Is there a ' + main_subj + ' in the image? Please answer only with a yes or no.'
  q2 = 'Is the ' + main_subj + ' ' + main_verb + '? Please answer only with a yes or no.'
  q3 = 'Is there a ' + sub_subj + ' in the image? Please answer only with a yes or no.'
  q4 = 'Is the ' + sub_subj + ' ' + sub_verb + ' the ' + main_subj + '? Please answer only with a yes or no.'
  if attribute1:
    q5 = 'Does the ' + sub_subj + ' have ' + attribute1 + '? Please answer only with a yes or no.'
  else: q5 = None
  if attribute2:
    q6 = 'Does the ' + sub_subj + ' have ' + attribute2 + '? Please answer only with a yes or no.'
  else: q6 = None
  LDD_df.loc[len(LDD_df)] = [prompt, main_subj, main_verb, sub_subj, sub_verb, attribute1, attribute2, q1, q2, q3, q4, q5, q6]

prompt_index = [i for i in range(1, N_PROMPTS+1) for _ in range(3)]
LDD_df['prompt_index'] = prompt_index
LDD_df.to_csv(f'{OUTPUT_DIR}/ldd_df.csv', index=False)

# AMBIGUITY questions
meanings = list(zip(amb_all[1::3], amb_all[2::3]))

ambiguity_df = pd.DataFrame(columns=['prompt', 'meaning1', 'meaning2', 'q1','q2','q3','q4','cq4'])
for i in range(len(meanings)):
  prompt = PROMPTS['Ambiguity'][i]
  meaning1 = meanings[i][0]
  meaning2 = meanings[i][1]
  q1 = 'Is this a good description of the image? ' + prompt + '. Please answer only with a yes or no.'
  q2 = 'Does the image fully represent the meaning of the following sentence? \"' + meaning1 + '\" Please answer only with a yes or no.'
  q3 = 'Does the image fully represent the meaning of the following sentence? \"' + meaning2 + '\" Please answer only with a yes or no.'
  q4 = 'Which of the following sentences better describes the image? 1. ' + meaning1 + '. 2. ' + meaning2 + '. Please answer only with the number associated with the sentence (1 or 2).'
  # control question (meaning 1 and meaning 2 are in reverse order)
  cq4 = 'Which of the following sentences better describes the image? 1. ' + meaning2 + '. 2. ' + meaning1 + '. Please answer only with the number associated with the sentence (1 or 2).'
  ambiguity_df.loc[len(ambiguity_df)] = [prompt, meaning1, meaning2, q1, q2, q3, q4, q5]

ambiguity_df.to_csv(f'{OUTPUT_DIR}/ambiguity_df.csv', index=False)