
## HDNO

This is the codebase for ICLR 2021 paper [Modelling Hierarchical Structure between Dialogue Policy and Natural Language Generator with Option Framework for Task-oriented Dialogue System](https://arxiv.org/abs/2006.06814).

If you use any source codes included in this toolkit in your work, please cite the following paper. The bibtex is listed below:     
```
@article{wang2020modelling,
  title={Modelling hierarchical structure between dialogue policy and natural language generator with option framework for task-oriented dialogue system},
  author={Wang, Jianhong and Zhang, Yuan and Kim, Tae-Kyun and Gu, Yunjie},
  journal={arXiv preprint arXiv:2006.06814},
  year={2020}
}
```

### Repository Structure
* configs: hyperparameters for different experiments
* data: dialogue dataset (MultiWowz 2.0 & MultiWoz 2.1)
* latent_dialog: source code
* visualize: visualize experiments result
* human_evaluator.py: to evalute groud truth's performance
* supervised.py: main entry for Supervised Learning
* reinforce.py: main entry for Reinforcement Learning


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

### Reproduce the result
For the convenience of reproducing the results shown in paper, we give simple bash scripts to do it.

1. preparing data
```
unzip data/multiwoz_2.0.zip -d data
unzip data/multiwoz_2.1.zip -d data
```

2. pretraining
```bash
sh train.sh sl woz2.0 # For MultiWoz 2.0
sh train.sh sl woz2.1 # For MultiWoz 2.1
```

3. hierarchical reinforcement learning (HRL)
```bash
sh train.sh rl woz2.0 # For MultiWoz 2.0
sh train.sh rl woz2.1 # For MultiWoz 2.1
```

4. evaluating trained model
```bash
sh test.sh sl woz2.0 5 # For MultiWoz 2.0 pretrained model
sh test.sh sl woz2.1 5 # For MultiWoz 2.1 pretrained model
sh test.sh rl woz2.0 2 # For MultiWoz 2.0 HRL model 
sh test.sh rl woz2.1 5 # For MultiWoz 2.1 HRL model
```

We have also released several trained models in the `model_save` folder, which can be directly evaluated to reproduce the results in paper. 

5. Also, we give a script that can train the model of HDNO based on the pretrained model we provide on MultiWoz 2.0 and MultiWoz 2.1:           
```bash
sh reproduce.sh
```

After training, run the following script to obtain the test results on the paper:
```bash
sh test.sh rl woz2.0 2 # For MultiWoz 2.0 HRL model 
sh test.sh rl woz2.1 5 # For MultiWoz 2.1 HRL model
```

### Main results

* The table shows the main test results of `HDNO` on MultiWoz 2.0 and MultiWoz 2.1 evaluated with the automatic evaluation metrics.       
![result](https://github.com/mikezhang95/HDNO/blob/master/visualize/result.png)

* The diagram demonstrates latent dialogue acts of `HDNO` clustered in 8 categories.    
![cluster](https://github.com/mikezhang95/HDNO/blob/master/visualize/cluster.png)







