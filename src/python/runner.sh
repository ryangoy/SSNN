#echo "running bed default.."
#python main.py --single_class='chair' > logs/schair.txt

#echo "running 1"
#python main.py --single_class='sofa' > logs/ssofa.txt

#echo "running 2"
#python main.py --single_class='board' > logs/sboard.txt

#echo "running 3"
#python main.py --single_class='table' > logs/stable.txt

# ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'nightstand', 'sofa', 'table', 'toilet']

echo "running toilet.."
python main.py --single_class='toilet' --output_category='toilet' > logs/toilet.txt

echo "running bathtub"
python main.py --single_class='bathtub' --output_category='bathtub' > logs/bathtub.txt

echo "running bookshelf"
python main.py --single_class='bookshelf' --output_category='bookshelf' > logs/bookshelf.txt

echo "running desk"
python main.py --single_class='desk'  --output_category='desk'> logs/desk.txt

echo "running dresser"
python main.py --single_class='dresser' --output_category='dresser' > logs/dresser.txt

echo "running bed"
python main.py --single_class='bed' --output_category='bed' --rotated_copies=1 > logs/bed.txt

echo "running chair"
python main.py --single_class='chair' --output_category='chair' --rotated_copies=1> logs/chair.txt

# echo "running nightstand"
# python main.py --single_class='nightstand' > logs/nightstand.txt

# echo "running sofa"
# python main.py --single_class='sofa' > logs/spfa.txt

# echo "running table"
# python main.py --single_class='table' > logs/table.txt

# echo "running toilet"
# python main.py --single_class='toilet' > logs/toilet.txt
