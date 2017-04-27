#!/usr/bin/env bash

DATASET_URL="http://157.253.63.7/lab9Detection.tar"
LABELS_URL="http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip"
LABELS_FILE="wider_face_split.zip"
DATASET_ROOT="lab9Detection"
DATASET_FILE="lab9Detection.tar"
VAL_URL="http://157.253.63.7/ValImagesLab9.zip"
VAL_FILE="ValImagesLab9.zip"
DATA_PATH="data"

MODEL_FILE="model.mat"

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
    rm $LABELS_FILE

    printf "Downloading validation dataset...\n"
    wget -c $VAL_URL
    unzip $VAL_FILE
    rm $VAL_FILE

    cd ..
    python process_labels.py
fi


VLFEAT_URL="http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz"
VLFEAT_FILE="vlfeat-0.9.20-bin.tar.gz"
VLFEAT_ORIGINAL_PATH="vlfeat-0.9.20"
VLFEAT_PATH="vlfeat"
printf "Checking if vlfeat has been already downloaded...\n"
if [ ! -d $VLFEAT_PATH ]; then
    printf "\nDownloading vlfeat...\n"
    wget -c $VLFEAT_URL
    tar -xvzf $VLFEAT_FILE
    rm $VLFEAT_FILE
    mv $VLFEAT_ORIGINAL_PATH $VLFEAT_PATH
fi

MATCONVNET_URL="http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta24.tar.gz"
MATCONVNET_FILE="matconvnet-1.0-beta24.tar.gz"
MATCONVNET_ORIGINAL_PATH="matconvnet-1.0-beta24"
MATCONVNET_PATH="matconvnet"
printf "Checking if MATCONVNET has been already downloaded...\n"
if [ ! -d $MATCONVNET_PATH ]; then
    printf "\nDownloading MATCONVNET...\n"
    wget -c $MATCONVNET_URL
    tar -xvzf $MATCONVNET_FILE
    rm $MATCONVNET_FILE
    mv $MATCONVNET_ORIGINAL_PATH $MATCONVNET_PATH
fi

if [! -f $MODEL_FILE]; then
    nohup /usr/local/matlab/bin/matlab -nodisplay -nojvm -nosplash -nodesktop -r "run('main');exit(0);" > out.log 2> err.log &
    printf $!
fi