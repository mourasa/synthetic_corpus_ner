#!/bin/bash

cd 3_generation/llama.cpp

while getopts 'r:hn:p:' OPTION; do
  case "$OPTION" in
    r)
      echo "Argument : $OPTARG"
      repo="$OPTARG"
      echo "Repository : ./results/$repo"
      ;;
    h)
      echo "you have supplied the -h option"
      ;;
    n)
      ntxt="$OPTARG"
      echo "Texts per prompts : $OPTARG"
      ;;
    p)
      nprompt="$OPTARG"
      echo "Prompts : $OPTARG"
      ;;
    ?)
      echo "script usage: $(basename \$0) [-l] [-h] [-n]" >&2
      exit 1
      ;;
  esac
done

shift "$(($OPTIND -1))"


function ProgressBar {
# Process data
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
# Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

printf "\r|  Progress : [${_fill// /#}${_empty// /-}] ${_progress}%% \n"

}

m=23
n=0
start=$(date +%k:%M:%S)
current_date=$(date +%s)
rm ../../results/"$repo"/log.txt
mkdir ../../4_annotation/generation/generation_$repo

for i in $(seq 0 $(($nprompt-1)))
do
	for k in $(seq 1 $ntxt)
	do
		echo "x--------------------- Text Generation ----------------------x" >> ../../results/"$repo"/log.txt
		echo "|  Current generation : $(($i+1))/$nprompt $k/$ntxt " >> ../../results/"$repo"/log.txt
		ProgressBar $(($(($i*$ntxt))+$k)) $(($nprompt*$ntxt)) >> ../../results/"$repo"/log.txt
		printf "|  Start Time : " >> ../../results/$repo/log.txt
		printf "$start \n" >> ../../results/$repo/log.txt
		printf "|  Time : " >> ../../results/"$repo"/log.txt
	        printf $(date +%k:%M:%S) >> ../../results/"$repo"/log.txt
		last_date=$current_date
		current_date=$(date +%s)
		printf "\n|  Duration of the last generation : $(($current_date - $last_date)) seconds \n" >> ../../results/"$repo"/log.txt
		n=$(($n+1))
		m=$(($m + $(($current_date - $last_date)) ))
		printf "|  Time average : $(($m/$n)) seconds \n" >> ../../results/"$repo"/log.txt
		t=$(($(($m*$(($(($nprompt*$ntxt))-$(($(($i*$ntxt))+$k))))))/$(($n))))
		printf "|  Estimated remaining time : $(($t/3600)) h $((($t%3600)/60)) min $(($t%60)) sec...\n" >> ../../results/$repo/log.txt
		echo "x------------------------------------------------------------x" >> ../../results/"$repo"/log.txt
		clear
                tail -n -9 ../../results/"$repo"/log.txt
		./main -m ../vigogne/models/7B_chat/ggml-model-q4_0.bin --color -c 2048 --temp 0.8 --top-k 40 --top-p 0.9 -r '."' -f ../../results/$repo/prompts/prompt_file_$i.txt -mg 1 -ngl 35 -n 500 > ./results/resultat_courant.txt 2>/dev/null
		tail -n -1 ./results/resultat_courant.txt | cut -c 15- | rev | cut -c 2- | rev >> ../../results/$repo/generation/generation_prompt_$i.txt
	done
done

cd ../..
