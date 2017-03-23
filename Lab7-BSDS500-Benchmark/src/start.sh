#! /usr/bin/env bash

DATA="http://157.253.63.7/BSDS500FastBench.tar.gz"
OUT_PATH="BSR"
FILE="BSDS500FastBench.tar.gz"

printf "Checking if dataset has been already downloaded...\n"
if [ ! -d $OUT_PATH ]; then
    # mkdir $OUT_PATH 2>/dev/null
    # cd $OUT_PATH
    printf "\nDownloading data...\n"
    wget -c $DATA
    tar -xvzf $FILE
    rm $FILE
    # cd ..
fi

python process.py