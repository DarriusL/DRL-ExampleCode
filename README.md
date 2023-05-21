# DRL-ExampleCode(Not finished, coding in progress)

code reference: [SLM_Lab](https://github.com/kengz/SLM-Lab)

Implementation code when learning deep reinforcement learning code(from SLM_Lab), only supported environments: CartPole.

## Environment configuration

```shell
git clone https://github.com/DarriusL/DRL-ExampleCode.git
cd DRL-ExampleCode
conda env create -f env.yml
conda activate dev
```





## Framework file structure

```

```





## Command

### usage

```shell
usage: executor.py [-h] [--config CONFIG] [--mode MODE] [--dev DEV]

options:
  -h, --help            show this help message and exit
  --config CONFIG, -cfg CONFIG
                        Path of configration.
  --mode MODE           Mode of operation.(train/test)
  --dev DEV             Enable code debugging
```



### qiuck start

reinforce

```shell
python executor.py -cfg='./config/reinforce/reinforce_cartpole.json' --mode='train'
python executor.py -cfg='./cache/data/reinforce/cartpole/[-opt-]/config.json' --mode='test'
```

sarsa

```shell
python executor.py -cfg='./config/sarsa/sarsa_cartpole.json' --mode='train'
python executor.py -cfg='./cache/data/sarsa/cartpole/[-opt-]/config.json' --mode='test'
```

