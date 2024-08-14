#!/bin/bash

# Check if the first parameter exists
if [ -z "$1" ]; then
  echo "No argument provided. Please provide an argument."
  exit 1
fi

# Use the first parameter as an argument
echo "The sample name  provided is: $1"
export SAMPLE=$1

ossutil cp -u --retry-times=20 --part-size 67108864 --parallel=8 --recursive subset_assembly.fa.gz oss://data-bucket-zjk/upload/${SAMPLE}/subset_assembly.fa.gz

echo $(date +"%Y-%m-%d-%H-%M-%S") finish uploading subset_assembly.fa.gz

# uploading sample pair files
ossutil cp -u --retry-times=20 --part-size 67108864 --parallel=8 --recursive ${SAMPLE}_1.fastq.gz oss://data-bucket-zjk/upload/${SAMPLE}/

echo $(date +"%Y-%m-%d-%H-%M-%S") finish uploading ${SAMPLE}_1.fastq.gz

ossutil cp -u --retry-times=20 --part-size 67108864 --parallel=8 --recursive ${SAMPLE}_2.fastq.gz oss://data-bucket-zjk/upload/${SAMPLE}/

echo $(date +"%Y-%m-%d-%H-%M-%S") finish uploading ${SAMPLE}_2.fastq.gz

