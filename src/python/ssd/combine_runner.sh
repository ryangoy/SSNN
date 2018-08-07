declare -a arr=("bathtub" "bed" "bookshelf" "chair" "desk" "dresser" "nightstand" "sofa" "table" "toilet")

for i in "${arr[@]}"
do
    echo "running combine for $i"
    DIRNAME=$i
    DIRNAME+="outputs"
    python combine_2d_3d.py /media/ryan/sandisk/SUNRGBD/$DIRNAME/ $i
done
