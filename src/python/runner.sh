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
python3 main.py --single_class='toilet' > logs/toilet.txt

echo "running bed"
python3 main.py --single_class='bed' > logs/bed.txt

echo "running 2"
python3 main.py --single_class='bathtub' > logs/bathtub.txt

echo "running 3"
python3 main.py --single_class='bookshelf' > logs/bookshelf.txt

echo "running 4"
python3 main.py --single_class='chair' > logs/chair.txt

echo "running 4"
python3 main.py --single_class='desk' > logs/desk.txt

echo "running 4"
python3 main.py --single_class='dresser' > logs/dresser.txt

echo "running 4"
python3 main.py --single_class='nightstand' > logs/nightstand.txt

echo "running 4"
python3 main.py --single_class='sofa' > logs/spfa.txt

echo "running 4"
python3 main.py --single_class='table' > logs/table.txt

echo "running 4"
python3 main.py --single_class='toilet' > logs/toilet.txt
