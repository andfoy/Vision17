#!/usr/bin/env bash

DATASET_URL="http://157.253.63.7/lab9Detection.tar"
LABELS_URL="http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip"
LABELS_FILE="wider_face_split.zip"
DATASET_ROOT="lab9Detection"
DATASET_FILE="lab9Detection.tar"
DATA_PATH="data"

printf "Checking if dataset has been already downloaded...\n"
if [ ! -d $DATA_PATH ]; then
    printf "\nDownloading dataset...\n"
    wget -c $DATASET_URL
    tar -xvf $DATASET_FILE
    printf "Uncompressing dataset...\n"
    rm $DATASET_FILE
    mv $DATASET_ROOT $DATA_PATH
    cd $DATA_PATH
    ZIPPED_FILES=$(ls *.zip)
    for ZIP_FILE in $ZIPPED_FILES; do
        # printf "..."
        unzip $ZIP_FILE
        rm $ZIP_FILE
    done
    printf "Done!\n"

    printf "nDownloading labels...\n"
    wget -c $LABELS_URL
    unzip $LABELS_FILE
    cd ..
    # python process_labels.py
fi

