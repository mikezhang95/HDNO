
## HDNO: Hieracrchical Structure of Dialogue Policy and Natural Language Genrator in Option Framework

### Requirements

1. install conda environment                                        
```bash
conda create -n hdno python=3.6
conda activate hdno
```

2.  install requirements
```bash
pip install -r requirements.txt     
```

### Repository Structure
* configs: hyperparameters for different experiments
* data: dialogue datset (MultiWowz 2.0 & MultiWoz 2.1)
* latent_dialog: source code
* visualize: visualize experiments result
* human_evaluator.py: to evalute groud truth's performance
* supervised.py: main entry for Supervised Learning
* reinforce.py: main entry for Reinforcement Learning
    

### Reproduce the result
1. prepare data
```
unzip data/multiwoz_2.0.zip -d data
unzip data/multiwoz_2.1.zip -d data
```

2. run Supervised Learning
```bash
sh train.sh sl woz2.0 # For MultiWoz 2.0
sh train.sh sl woz2.1 # For MultiWoz 2.1
```

3. run Reinforcement Learning
```bash
sh train.sh rl woz2.0 # For MultiWoz 2.0
sh train.sh rl woz2.1 # For MultiWoz 2.1
```

4. evaluate trained model
```bash
sh test.sh sl woz2.0 # For MultiWoz 2.0 SL model
sh test.sh sl woz2.1 # For MultiWoz 2.1 SL model
sh test.sh rl woz2.0 # For MultiWoz 2.0 RL model
sh test.sh rl woz2.1 # For MultiWoz 2.1 RL model
```
