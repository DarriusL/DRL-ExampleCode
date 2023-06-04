# DRL-ExampleCode

code reference: [SLM_Lab](https://github.com/kengz/SLM-Lab)

Implementation code when learning deep reinforcement learning code(from SLM_Lab mainly).

## Environment configuration

```shell
git clone https://github.com/DarriusL/DRL-ExampleCode.git
cd DRL-ExampleCode
conda env create -f env.yml
conda activate dev
```





## Framework file structure

```
├── .gitignore
├── agent
│	├── algorithm
│	│	├── actor_critic.py
│	│	├── alg_util.py
│	│	├── base.py
│	│	├── dqn.py
│	│	├── reinforce.py
│	│	├── sarsa.py
│	│	└── __init__.py
│	├── memory
│	│	├── base.py
│	│	├── offpolicy.py
│	│	├── onpolicy.py
│	│	└── __init__.py
│	└── net
│		├── base.py
│		├── mlp.py
│		├── net_util.py
│		└── __init__.py
├── config
│	├── a2c
│	│	└── a2c_nstep_cartpole_on.json
│	├── dqn
│	│	├── doubledqn_cartpole_off.json
│	│	├── doubledqn_cartpole_per.json
│	│	├── dqn_cartpole_off.json
│	│	└── targetdqn_cartpole_off.json
│	├── lab_cfg.json
│	├── reinforce
│	│	├── reinforce_cartpole_mc.json
│	│	├── reinforce_entropyreg_cartpole_mc.json
│	│	└── reinforce_entropyreg_cartpole_nstep.json
│	└── sarsa
│		├── sarsa_cartpole_mc.json
│		└── sarsa_cartpole_nstep.json
├── env
│	├── base.py
│	├── openai_gym.py
│	└── __init__.py
├── env.yml
├── executor.py
├── lib
│	├── callback.py
│	├── glb_var.py
│	├── json_util.py
│	└── util.py
├── LICENSE
├── README.md
└── room
	├── system
	│	├── base.py
	│	├── offpolicy.py
	│	├── onpolicy.py
	│	└── __init__.py
	└── work.py

```





## Command

### usage

```shell
usage: executor.py [-h] [--config CONFIG] [--mode MODE] [--dev DEV]
```

### option

```shell
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
python executor.py -cfg='./config/reinforce/reinforce_cartpole_mc.json' --mode='train'
python executor.py -cfg='./config/reinforce/reinforce_entropyreg_cartpole_mc.json' --mode='train'
python executor.py -cfg='./config/reinforce/reinforce_entropyreg_cartpole_nstep.json' --mode='train'
python executor.py -cfg='./cache/data/reinforce/cartpole/[-opt-]/config.json' --mode='test'
```

sarsa

```shell
python executor.py -cfg='./config/sarsa/sarsa_cartpole_nstep.json' --mode='train'
python executor.py -cfg='./config/sarsa/sarsa_cartpole_mc.json' --mode='train'
python executor.py -cfg='./cache/data/sarsa/cartpole/[-opt-]/config.json' --mode='test'
```

dqn

```shell
python executor.py -cfg='./config/dqn/dqn_cartpole_off.json' --mode='train'
python executor.py -cfg='./config/dqn/targetdqn_cartpole_off.json' --mode='train'
python executor.py -cfg='./config/dqn/doubledqn_cartpole_off.json' --mode='train'
python executor.py -cfg='./config/dqn/doubledqn_cartpole_per.json' --mode='train'
```

a2c

```shell
python executor.py -cfg='./config/a2c/a2c_nstep_cartpole_on.json' --mode='train'
```

