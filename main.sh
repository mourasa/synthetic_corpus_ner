#!/bin/bash

file='main_2000_2'
np=2000
nt=1
rm -r ./results/"$file"
mkdir ./results/"$file"

python ./1_matrix/create_matrix.py "$np"
cp ./2_prompt/datasets/matrix.csv ./results/"$file"

python ./2_prompt/prompt.py
mkdir ./results/"$file"/prompts
cp ./2_prompt/prompts/* ./results/"$file"/prompts

mkdir ./results/"$file"/generation
bash 3_generation/llama.cpp/generation.sh -r "$file" -n "$nt" -p "$np"

python ./4_annotation/corpus_filtred.py "$file" "$np"
