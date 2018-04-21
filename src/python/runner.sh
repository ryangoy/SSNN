echo "running default.."
python main.py --single_class='sofa' --num_epochs=50 > logs/sofa.txt

echo "running 1"
python main.py --single_class='table' > logs/table.txt

echo "running 2"
python main.py --single_class='chair' > logs/chair.txt

echo "running 4"
python main.py --single_class='board' > logs/board.txt

