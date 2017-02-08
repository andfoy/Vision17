#! /usr/bin/env bash

OPT="$1"
if [[ ($OPT = "-h") || ($OPT = "") ]]; then
    echo "Display basic image info: Filename, Format and Display dimensions"
    echo ""
    echo "Usage:"
    echo "  ./image_info.sh -h"
    echo "  ./image_info.sh <dir>"
    echo ""
    echo "Options:"
    echo "  -h  Show this screen"
    echo ""
    echo "Arguments:"
    echo "  dir: Path to the folder that contains the BSDS500 data set (BSR folder)"
    exit 1
fi

# cd $OPT

IMG_PATH="$OPT/BSDS500/data/images"
PARTITIONS=$(ls $IMG_PATH)
# echo $PARTITIONS
PORTRAIT=0
LANDSCAPE=0

for part in $PARTITIONS; do
    CUR_PATH="$IMG_PATH/$part"
    for file in $(ls $CUR_PATH/*.jpg); do
        # echo $file
        IMG_INFO=$(identify $file)
        DIMS=$(echo $IMG_INFO | awk '{print $3}')
        # echo $DIMS
        # echo $IMG_INFO
        DIMARR=(${DIMS//x/ })
        W=${DIMARR[0]}
        H=${DIMARR[1]}
        ORIENTATION="Landscape"
        if [[ $H > $W ]]; then
            ORIENTATION="Portrait"
            PORTRAIT=$((PORTRAIT+1))
        else
            LANDSCAPE=$((LANDSCAPE+1))
        fi
        OUT_STR=$(echo $IMG_INFO | awk '{print $1, $2, $3}')
        echo "$OUT_STR $ORIENTATION"
    done
done

echo "Number of Landscape images: $LANDSCAPE"
echo "Number of Portrait images: $PORTRAIT"