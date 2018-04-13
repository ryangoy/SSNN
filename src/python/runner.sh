echo "running default.."
python main.py > logs/default.txt

echo "running 1"
python main.py --load_from_npy=True --k_size_factor=1 > logs/k1.txt

echo "running 2"
python main.py --load_from_npy=True --num_kernels=16 > logs/nk16.txt

echo "running 3"
python main.py --load_from_npy=True --learning_rate=0.001 > logs/lr001.txt

echo "running 4"
python main.py --load_from_npy=True --num_dot_layers=32 > logs/ndl32.txt

