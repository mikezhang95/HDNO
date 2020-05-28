## Rethinking Action Spaces for Reinforcement Learning in End-to-end Dialog Agents with Latent Variable Models

### config

    | Parameters           | Explanation                          | sl_cat |
    |----------------------|--------------------------------------|--------|
    | max_utt_len          | max length of encoder                | 50     |
    | max_dec_len          | max length of decoder                | 50     |
    | backward_size        | context comes from past (bs) turn    | 2      |
    | embed_size           | word embedding dimension             | 100    |
    | num_layers           |                                      | 1      |
    | y_size               | dim of latent variable               | 10     |
    | k_size               | num of class for catergorical latent | 20     |
    | beta                 |                                      | 20     |
    | simple_posterior     | Lite ELBO                            | True   |
    | contextual_posterior |                                      | True   |
    | use_mi               |                                      | False  |
    | use_pr               |                                      | True   |
    | use_diversity        |                                      | False  |
    | beam_size            | beam seach width                     | 20     |
    | fix_batch            | every session is a batch if True     | True   |
    | fix_train_batch      | num_batch*batch_size data if False   | False  |
    | avg_type             |                                      | "word" |
    | improve_threshold    |                                      | 0.996  |
    | patient_increase     |                                      | 2.0    |
    | early_stop           | early stop without improvement       | False  |
    | preview_batch_num    |                                      | null   |
    | init_range           |                                      | 0.1    |
    | forward_only         | without training and saving models   | False  |
    | print_step           | print train loss every {} batch      | 300    |



### Data
The data are in folder data. For DealOrNoDeal dataset, the files are in data/negotiate. For MultiWoz dataset,
the processed version is a zip file (norm-multi-woz.zip). Please unzip it before run any experiments for MultiWoz.

            
### Over structure:
The source code is under **latent_dialog**. The experiment script is under folders:

    - experiments_deal: scripts for studies on DealOrNoDeal
    - experiments_woz: scripts for studies on MultiWoz
    
For both datasets, the scripts follow the same structure: (1) first using supervised learning
to create pre-train models. (2) use policy gradient reinforcement learning to fine tune the pretrain
model via reinforcement learning.

Besides that, the other folders contains:
    
    - FB: the original facebook implementation from Lewis et al 2017. We the pre-trained judge model 
    to score our DealOrNoDeal conversations.
    - latent_dialog: source code 

### Step 1: Supervised Learning

    - sl_word: train a standard encoder decoder model using supervised learning (SL)
    - sl_cat: train a latent action model with categorical latetn varaibles using SL.
    - sl_gauss: train a latent action model with gaussian latent varaibles using SL.

### Step 2: Reinforcement Learning
Set the system model folder path in the script:
       
    folder = '2019-04-15-12-43-05-sl_cat'
    epoch_id = '8'
    
And then set the user model folder path in the script
    
    sim_epoch_id = '5'
    simulator_folder = '2019-04-15-12-43-38-sl_word'  # set to the log folder of the user model

Each script is used for:

    - reinforce_word: fine tune a pretrained model with word-level policy gradient (PG)
    - reinforce_cat: fine tune a pretrained categorical latent action model with latent-level PG.
    - reinforce_gauss: fine tune a pretrained gaussian latent action model with latent-level PG.
