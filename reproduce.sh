
mkdir outputs

# reproduce multiwoz 2.0
mkdir outputs/rl_hdno_woz2.0
cp model_save/woz2.0/alpha_0.0001/model  outputs/rl_hdno_woz2.0/1000-model
sh test.sh rl woz2.0 2

# reproduce multiwoz 2.0
mkdir outputs/rl_hdno_woz2.1
cp model_save/woz2.1/alpha_0.01/model  outputs/rl_hdno_woz2.1/1000-model
sh test.sh rl woz2.1 5
