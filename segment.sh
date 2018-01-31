#!/bin/bash


if [ "$#" -ne 3 ]; then
    echo "3 arguments must be provided"
		python infer.py -h
		exit
fi


MODEL=$1
FOLDER=$PWD/$2
OUT=$3
for img in `ls $FOLDER`;
do
	echo "Segmenting image : $img" 
	python infer.py $MODEL $FOLDER/$img --out $OUT &> /dev/null
done
 
