#! /usr/bin/env bash

DIR="$1"
if [[ ($DIR = "") || ($DIR = "-h") ]]; then
    echo "Find image duplicates"
    echo ""
    echo "Usage:"
    echo "  ./duplicates.sh -h"
    echo "  ./duplicates.sh <dir>"
    echo ""
    echo "Options:"
    echo "  -h  Show this screen"
    echo ""
    echo "Arguments:"
    echo "  dir: Path to a folder that contains images"
    exit 1
fi

IMG_EXT="*.png *.jpg *.tiff *.svg *.bmp *.gif *.raw"

declare -A dupl

# SAVEIFS=$IFS
# IFS=$(echo -en "\n\b")

for ext in $IMG_EXT; do
    FILE_LIST=$(ls $DIR/$ext 2>/dev/null )
    OUT=$?
    if [[ $OUT = "0" ]]; then
        for f in $FILE_LIST; do
            # echo $f
            CHKSUM=$(md5sum $f | awk '{print $1}')
            # echo $CHKSUM
            if [[ -v ${dupl[$CHKSUM]} ]]; then
                echo $f
            else
                dupl[$CHKSUM]=f
            fi
        done
    fi
done

# IFS=$SAVEIFS