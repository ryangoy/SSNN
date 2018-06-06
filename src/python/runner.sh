#declare -a arr=("bathtub" "bed" "bookshelf" "chair" "desk" "dresser" "nightstand" "sofa" "table" "toilet")
declare -a arr=("chair" "night_stand")

for i in "${arr[@]}"
do
    echo "running $i"
    python main.py --single_class=$i --output_category=$i --checkpoint_save_dir=/media/ryan/sandisk/SUNRGBD/checkpoints/$i > logs/$i.txt
done
