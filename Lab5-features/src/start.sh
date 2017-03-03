#! /usr/bin/env bash

DATA="http://157.253.63.7/textures.tar.gz"
OUT_PATH="data"
FILE="textures.tar.gz"

printf "Checking if dataset has been already downloaded...\n"
if [ ! -d data ]; then
    mkdir $OUT_PATH 2>/dev/null
    cd $OUT_PATH
    printf "\nDownloading data...\n"
    wget -c $DATA
    tar -xvzf $FILE
    rm $FILE
    cd ..
fi

printf "\nCompiling Cython modules...\n\n"

cd textons
python setup.py build_ext -i
