echo "running bed default.."
python main.py --single_class='chair' > logs/schair.txt

echo "running 1"
python main.py --single_class='sofa' > logs/ssofa.txt

echo "running 2"
python main.py --single_class='board' > logs/sboard.txt

echo "running 3"
python main.py --single_class='table' > logs/stable.txt

