
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

mode=$1
dataset=$2

if [ $mode == "sl" ]; then
  python -u supervised.py --config_name sl_hdno_$dataset
elif [ $mode == "rl" ]; then
  python -u reinforce.py --config_name rl_hdno_$dataset
else
  echo "wrong input"
fi
