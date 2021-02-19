
mkdir outputs

# reproduce multiwoz 2.0 (pretraining)
mkdir outputs/sl_hdno_woz2.0
cp model_save/woz2.0/pretrain/model  outputs/sl_hdno_woz2.0/1000-model
sh test.sh sl woz2.0 1
printf "woz2.0 sl done \n\n"

# reproduce multiwoz 2.0 (sync)
mkdir outputs/rl_hdno_woz2.0
cp model_save/woz2.0/sync/model  outputs/rl_hdno_woz2.0/1000-model
sh test.sh rl woz2.0 5
printf "woz2.0 rl(sync) done \n\n"

# reproduce multiwoz 2.0 (async)
mkdir outputs/rl_hdno_woz2.0
cp model_save/woz2.0/alpha_0.0001/model  outputs/rl_hdno_woz2.0/1000-model
sh test.sh rl woz2.0 2
printf "woz2.0 rl(async) done \n\n"


# reproduce multiwoz 2.1 (pretraining)
mkdir outputs/sl_hdno_woz2.1
cp model_save/woz2.1/pretrain/model  outputs/sl_hdno_woz2.1/1000-model
sh test.sh sl woz2.1 5
printf "woz2.1 sl done \n\n"

# reproduce multiwoz 2.1 (sync)
mkdir outputs/rl_hdno_woz2.1
cp model_save/woz2.1/sync/model  outputs/rl_hdno_woz2.1/1000-model
sh test.sh rl woz2.1 5
printf "woz2.1 rl(sync) done \n\n"

# reproduce multiwoz 2.0 (async)
mkdir outputs/rl_hdno_woz2.1
cp model_save/woz2.1/alpha_0.01/model  outputs/rl_hdno_woz2.1/1000-model
sh test.sh rl woz2.1 5
printf "woz2.1 rl(async) done \n\n"
