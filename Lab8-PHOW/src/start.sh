#!/usr/bin/env bash

DATA_PATH="data"
VLFEAT_URL="http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz"
VLFEAT_FILE="vlfeat-0.9.20-bin.tar.gz"
VLFEAT_PATH="vlfeat-0.9.20"
printf "Checking if vlfeat has been already downloaded...\n"
if [ ! -d $VLFEAT_PATH ]; then
    printf "\nDownloading vlfeat...\n"
    wget -c $VLFEAT_URL
    tar -xvzf $VLFEAT_FILE
    rm $VLFEAT_FILE
fi

printf "\nCreating data folder....\n"
mkdir $DATA_PATH 2>/dev/null
# cd $DATA_PATH

printf "Checking if Caltech-101 dataset has been already downloaded...\n"
CALTECH_URL="http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
CALTECH_FILE="101_ObjectCategories.tar.gz"
CALTECH_PATH="$DATA_PATH/caltech-101"

if [ ! -d $CALTECH_PATH ]; then
    printf "\nDownloading Caltech-101...\n"
    mkdir $CALTECH_PATH
    cd $CALTECH_PATH
    wget -c $CALTECH_URL
    tar -xvf $CALTECH_FILE
    rm $CALTECH_FILE
    cd ../..
fi

printf "Checking if ImageNet subset has been already downloaded...\n"
IMAGENET_URL="http://157.253.63.7/imageNet200.tar"
IMAGENET_FILE="imageNet200.tar"
IMAGENET_PATH="$DATA_PATH/imagenet"

if [ ! -d $IMAGENET_PATH ]; then
    printf "\nDownloading ImageNet...\n"
    mkdir $IMAGENET_PATH
    cd $IMAGENET_PATH
    wget -c $IMAGENET_URL
    tar -xvf $IMAGENET_FILE
    rm $IMAGENET_FILE
    cd ../..
fi
