echo "1"
python main.py --single_class=bed --learning_rate=0.0005
echo "2"
python main.py --single_class=bed --loc_loss_lambda=1
echo "3"
python main.py --single_class=bed --num_kernels=4
echo "4"
python main.py --single_class=bed --num_anchors=4
echo "5"
python main.py --single_class=bed --rotated_copies=3
