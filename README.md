# Taxi Q-Learning Project

This project uses Q-learning to train an agent to complete the Taxi-v4 environment in Gymnasium. The goal is for the taxi to pick up a passenger and drop them off at the correct location while earning as much reward as possible.

## How to Run

Make sure Python is installed, then install the required libraries:

```bash
pip install -r requirements.txt
pip install "gymnasium[toy-text]"
```

Run the program with:

```bash
python train.py --episodes 12000
```

## Experiments

I tested how changing different Q-learning parameters affected the agent's performance:

- Learning rate: 0.1, 0.5, 0.9
- Discount factor: 0.5, 0.9, 0.99
- Epsilon decay: 0.999, 0.9995, 0.9999

Each configuration was trained for 12,000 episodes. I compared the results using reward graphs and evaluated the trained agents over 100 episodes.

## Results

The agent reached a 100% success rate during evaluation, with an average reward of 7.9 and an average of 13.1 steps per episode.

Higher learning rates generally helped the agent learn faster. Epsilon decay had the biggest effect on how quickly the training rewards improved.

## Files

- `train.py` — Q-learning code and experiments
- `results/` — graphs and evaluation results
- `videos/` — recording of the trained taxi
- `requirements.txt` — required Python libraries

The graphs show how the agent improved throughout training and how the different parameter settings affected learning.