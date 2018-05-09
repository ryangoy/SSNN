
echo "running 1 anchor"
python3 main.py --data_dir=/home/ryan/buildings --dataset_name=stanford --num_anchors=1 > logs/anchor1.txt

echo "running 4 anchors"
python3 main.py --data_dir=/home/ryan/buildings --dataset_name=stanford --num_anchors=4 > logs/anchor4.txt

echo "running 7 anchors"
python3 main.py --data_dir=/home/ryan/buildings --dataset_name=stanford --num_anchors=7 > logs/anchor7.txt

echo "running 9 anchors"
python3 main.py --data_dir=/home/ryan/buildings --dataset_name=stanford --num_anchors=9 --num_epochs=200 > logs/anchor9.txt

