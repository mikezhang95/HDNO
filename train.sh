
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python -u supervised.py --config_name sl_gendisclite_woz2.0 
# python -u reinforce.py --config_name rl_gendisclite_woz2.0 
