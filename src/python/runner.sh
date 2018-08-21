declare -a arr=("bathtub" "bed" "bookshelf" "desk" "dresser" "nightstand" "sofa" "table" "toilet")
#declare -a arr=("chair" "desk" "dresser" "nightstand" "sofa" "table" "toilet")

#declare -a arr=("sofa" "desk" "table")

for i in "${arr[@]}"
do
    echo "running $i"
    python main.py --single_class=$i --rotated_copies=3 --output_category=$i > logs/$i.txt
done

echo "running chair"
python main.py --single_class=chair --rotated_copies=0 --output_category=chair > logs/chair.txt
