echo "1"
python main.py --load_probe_output=False --load_from_npy=False --checkpoint_save_dir=/home/ryan/cs/datasets/SSNN/checkpoints/ --single_class=bed > logs/bed.txt

echo "2"
python main.py --load_probe_output=False --load_from_npy=False --checkpoint_save_dir=/home/ryan/cs/datasets/SSNN/checkpoints/ --single_class=toilet > logs/toilet.txt

echo "3"
python main.py --load_probe_output=False --load_from_npy=False --checkpoint_save_dir=/home/ryan/cs/datasets/SSNN/checkpoints/ --single_class=chair > logs/chair.txt

echo "4"
python main.py --load_probe_output=False --load_from_npy=False --checkpoint_save_dir=/home/ryan/cs/datasets/SSNN/checkpoints/ --single_class=bed > logs/bed.txt

echo "5"
python main.py --load_probe_output=False --load_from_npy=False --checkpoint_save_dir=/home/ryan/cs/datasets/SSNN/checkpoints/ --single_class=bookshelf > logs/bookshelf.txt
