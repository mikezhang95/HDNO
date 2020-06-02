
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

mode=$1
dataset=$2
alias=$3

if [ $mode == "sl" ]; then
  python -u supervised.py --config_name sl_hdno_$dataset --alias $alias --forward_only --gen_type beam --beam_size $4
elif [ $mode == "rl" ]; then
  python -u reinforce.py --config_name rl_hdno_$dataset --alias $alias --forward_only --gen_type beam --beam_size $4
else
  echo "wrong input"
fi
