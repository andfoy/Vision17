#! /usr/bin/env bash

DATA="http://157.253.63.7/textures.tar.gz"
OUT_PATH="data"
FILE="tar -xvzf textures.tar.gz"
# mkdir $OUT_PATH 2>/dev/null
# cd $OUT_PATH
echo "Checking if dataset has been already downloaded...\n"
if [ ! -d data ]; then
    mkdir $OUT_PATH 2>/dev/null
    cd $OUT_PATH
    echo "Downloading data...\n"
    wget -c $DATA
    tar -xvzf $FILE
    rm $FILE
    cd ..
fi
