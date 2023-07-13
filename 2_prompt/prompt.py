import pandas as pd
import os
import sys

os.chdir('/home/getalp/mourasa/pipeline/2_prompt')

df=pd.read_csv('./datasets/matrix.csv')

prompt_1='./datasets/prompt_template.txt'

# Read the prompt template
with open(prompt_1, 'r', encoding='UTF-8') as prompt_file:
  prompt=prompt_file.read()
  #print(prompt)

# Loop through the df and extract the relevant info
for index, row in df.iterrows():
  person=row['PERSON']
  organization=row['ORGANIZATION']
  place=row['PLACE']

  new_prompt=prompt

  # Replace missing info with variable values
  for r in (("PERSON", person), ("ORGANIZATION", organization), ("PLACE", place)):
    new_prompt = new_prompt.replace(*r)

  # Write new prompt files
  new_prompt_file=f'./prompts/prompt_file_{index}.txt'
  with open(new_prompt_file, 'w', encoding='UTF-8') as writer:
    writer.write(new_prompt)
   # writer.write('\n<ASSISTANT>:')
