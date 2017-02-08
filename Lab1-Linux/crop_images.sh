#! /usr/bin/env bash

opt()
{
    echo "Crop images"
    echo ""
    echo "Usage:"
    echo "  ./crop_images.sh --help"
    echo "  ./crop_images.sh -w <width> -h <height> <dir> <out_dir>"
    echo ""
    echo "Options:"
    echo "  --help  Show this screen"
    echo "  -w  Output image width"
    echo "  -h  Output image height"
    echo ""
    echo "Arguments:"
    echo "  dir: Path to the folder that contains the BSDS500 data set (BSR folder)"
    echo "  out_dir: Path to the folder where the cropped images are to be saved"
}

WIDTH="256"
HEIGHT="256"
SET_PATH=""
OUT_PATH=""

OPT1="$1"
if [[ ($OPT1 = "--help") || ($OPT1 = "") ]]; then
    opt
    exit 1
elif [[ $OPT1 = "-w" ]]; then
    WIDTH="$2"
    if [[ "$3" = "-h" ]]; then
        HEIGHT="$4"
        SET_PATH="$5"
        OUT_PATH="$6"
    else
        SET_PATH="$3"
        OUT_PATH="$4"
    fi
elif [[ $OPT1 = "-h" ]]; then
    HEIGHT="$2"
    if [[ "$3" = "-w" ]]; then
        WIDTH="$4"
        SET_PATH="$5"
        OUT_PATH="$6"
    else
        SET_PATH="$3"
        OUT_PATH="$4"
    fi
else
    opt
    exit 1
fi

echo $WIDTH
echo $HEIGHT
echo $SET_PATH
echo $OUT_PATH

mkdir $OUT_PATH 2>/dev/null

IMG_PATH="$SET_PATH/BSDS500/data/images"
PARTITIONS=$(ls $IMG_PATH)

for part in $PARTITIONS; do
    CUR_PATH="$IMG_PATH/$part"
    CUR_CROP="$OUT_PATH/$part"
    mkdir $CUR_CROP 2>/dev/null
    for file in $(ls $CUR_PATH/*.jpg); do
        filename=$(basename $file)
        out_filename="$CUR_CROP/$filename"
        crop_dim="$WIDTH x $HEIGHT+0+0"
        crop_dim=(${crop_dim// x /x})
        echo $out_filename
        convert $file -gravity Center -crop $crop_dim +repage $out_filename
    done
done