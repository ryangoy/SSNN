echo "running bed default.."
python main.py --single_class='bed' > logs/default.txt

echo "running 1"
python main.py --single_class='bed' --loc_loss_lambda=5 > logs/ll5.txt

echo "running 2"
python main.py --single_class='bed' --loc_loss_lambda=0.5 > logs/llp5.txt

echo "running 3"
python main.py --single_class='bed' --learning_rate=0.0001 > logs/hlr.txt

echo "running 4"
python main.py --single_class='bed' --learning_rate=0.00001 > logs/llr.txt
