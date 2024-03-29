# BoosterLander

- [BoosterLander](#boosterlander)
  - [Getting Started](#getting-started)
    - [Install](#install)
    - [Run](#run)
  - [Training](#training)
  - [Evaluation](#evaluation)

https://user-images.githubusercontent.com/35966693/151722087-3786d91f-ed2a-4159-8ecb-22ca0d321a34.mov

BoosterLander is an OpenAI environment based on [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/). The environment is intended to simulate the landing of a SpaceX Falcon 9 rocket. See the full dissertation based on this project [here](https://github.com/CameronFleet/final-year-project/files/7967167/main.pdf). Three variations of this environment exist:

  - BoosterLander (standard environment, no random noise simulating wind etc)
  - NoisyBoosterLander (environment with more noise, e.g. wind)
  - BrokenBoosterLander (environment where booster enters various sensor failure modes)

## Getting Started

### Install
```sh
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

### Run
Run the 'evaluation' using the provided pre-trained weights to test the install has worked as expected
```python
python evaluation.py -env-n -e DDQN -p algorithms/rl/weights/best/ddqn/BEST_248
```

## Training

Both Q-learning and Sarsa training scripts exist

To train a DDQN on the BoosterLander environment.
```sh
$ cd algorithms/rl
$ python -m q_learning
```

To run Q-learning on the NoistBoosterLander environment, train a SGD model
```sh
$ python -m q_learning -n -e SGD --save-dir MY_SAVE_DIR
```

To run SARSA on the BoosterLander environment, training a SGD model. 
```sh
$ python -m sarsa
```

## Evaluation
Evaluation allows you to test the pre-trained models in the different environment. The models below were trained over the course of weeks on a unviersity computer cluster.

To evaluate a PID controller for BoosterLander
```sh
$ python evaluation.py 
```

To evaluate a PID controller for NoisyBoosterLander
```sh
$ python evaluation.py -env-n 
```

To evaluate a SGD estimator (trained with Sarsa) with the best trained model for NoisyBoosterLander
```sh
$ python evaluation.py -env-n -e SGD -p algorithms/rl/weights/best/sgd_sarsa/BEST_231
```

To evaluate a SGD estimator (trained with Q-learning) with the best trained model for NoisyBoosterLander
```sh
$ python evaluation.py -env-n -e SGD -p algorithms/rl/weights/best/sgd_q_learning/BEST_218
```

To evaluate a DDQN estimator with the best trained model for NoisyBoosterLander
```sh
$ python evaluation.py -env-n -e DDQN -p algorithms/rl/weights/best/ddqn/BEST_248
```

To evaluate a DDQN estimator with the best trained model for the accelerometer broken case of BrokenBoosterLander run as below
> In config set ACCELEROMETER_FAILURE_CHANCE = 1
```sh
$ python evaluation.py -env-b -e DDQN -p algorithms/rl/weights/best/ddqn_acc_broken/BoosterLander_11000
```

To evaluate a DDQN estimator with the best trained model for the GPS broken case of BrokenBoosterLander run as below
> In config set GPS_FAILURE_CHANCE = 1
```sh
$ python evaluation.py -env-b -e DDQN -p algorithms/rl/weights/best/ddqn_gps_broken/BoosterLander_9999
```

To evaluate a DDQN estimator with the best trained model for the roll rate broken case of BrokenBoosterLander run as below
> In config set ROLL_FAILURE_CHANCE = 1
```sh
$ python evaluation.py -env-b -e DDQN -p algorithms/rl/weights/best/ddqn_rr_broken/BoosterLander_8000
```

To evaluate a DDQN estimator with the best trained model for the thrust broken case of BrokenBoosterLander run as below
> In config set SIDE_BOOSTER_FAILURE_CHANCE = 1
```sh
$ python evaluation.py -env-b -e DDQN -p algorithms/rl/weights/best/ddqn_thrust_broken/BEST_277
```
