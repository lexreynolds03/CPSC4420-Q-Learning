"""CPSC 4420 HW2: original tabular Q-learning on Gymnasium Taxi-v4.
Run: python train.py --episodes 12000
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
VIDEOS = ROOT / 'videos'

def train(alpha=.5, gamma=.95, eps_start=1., eps_end=.05, eps_decay=.9995,
          episodes=12000, seed=42):
    env = gym.make('Taxi-v4')
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rewards, lengths, successes = [], [], []
    epsilon = eps_start
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total, steps, success = 0., 0, False
        for step in range(200):
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                candidates = np.flatnonzero(q[state] == q[state].max())
                action = int(rng.choice(candidates))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            target = reward if done else reward + gamma * np.max(q[next_state])
            q[state, action] += alpha * (target - q[state, action])
            total += reward
            steps = step + 1
            if terminated and reward == 20:
                success = True
            state = next_state
            if done:
                break
        rewards.append(total)
        lengths.append(steps)
        successes.append(int(success))
        epsilon = max(eps_end, epsilon * eps_decay)
    env.close()
    return q, np.array(rewards), np.array(lengths), np.array(successes)

def evaluate(q, seed=100000, episodes=100):
    env = gym.make('Taxi-v4')
    rewards, lengths, successes = [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total, success = 0., False
        for step in range(200):
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated and reward == 20:
                success = True
            if terminated or truncated:
                break
        rewards.append(total)
        lengths.append(step + 1)
        successes.append(int(success))
    env.close()
    return {'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': float(np.mean(successes)),
            'mean_steps': float(np.mean(lengths)), 'eval_episodes': episodes}

def smooth(values, window=200):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window)/window, mode='valid')

def save_plot(series, title, filename):
    plt.figure(figsize=(9, 5))
    for label, rewards in series.items():
        smoothed = smooth(rewards)
        plt.plot(np.arange(len(smoothed)) + 200, smoothed, label=label)
    plt.xlabel('Training episode')
    plt.ylabel('Mean reward (200-episode moving average)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=170)
    plt.close()

def record_video(q):
    import imageio.v2 as imageio
    env = gym.make('Taxi-v4', render_mode='rgb_array')
    frames = []
    # Find a solvable initial state for a clear demonstration.
    for seed in range(200000, 200100):
        state, _ = env.reset(seed=seed)
        current_frames = [env.render()]
        success = False
        for _ in range(200):
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            current_frames.append(env.render())
            if terminated or truncated:
                success = terminated and reward == 20
                break
        if success:
            frames = current_frames
            break
    env.close()
    if not frames:
        raise RuntimeError('No successful episode found for video; retrain or adjust parameters.')
    # Macroblock-friendly even dimensions.
    imageio.mimsave(VIDEOS / 'trained_taxi.mp4', frames, fps=3, macro_block_size=1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=12000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    VIDEOS.mkdir(exist_ok=True)
    base = dict(alpha=.5, gamma=.95, eps_start=1., eps_end=.05,
                eps_decay=.9995, episodes=args.episodes, seed=args.seed)
    experiments = {
        'baseline': [('baseline', {})],
        'learning_rate': [(f'alpha={x}', {'alpha': x}) for x in (.1, .5, .9)],
        'discount_factor': [(f'gamma={x}', {'gamma': x}) for x in (.5, .9, .99)],
        'epsilon_decay': [(f'decay={x}', {'eps_decay': x}) for x in (.999, .9995, .9999)]
    }
    rows, candidates = [], []
    for group, variants in experiments.items():
        series = {}
        for label, changes in variants:
            config = dict(base, **changes)
            q, rewards, lengths, successes = train(**config)
            metrics = evaluate(q)
            row = dict(experiment=group, setting=label, **changes,
                       train_last_1000_reward=float(np.mean(rewards[-1000:])),
                       train_last_1000_success=float(np.mean(successes[-1000:])), **metrics)
            rows.append(row)
            series[label] = rewards
            candidates.append((metrics['mean_reward'], label, q, config, metrics))
            print(f'{group}: {label} => {metrics}', flush=True)
        save_plot(series, group.replace('_',' ').title(), group + '.png')
    # Selection uses the same held-out evaluation seeds for fair comparisons.
    best = max(candidates, key=lambda item: (item[0], item[4]['success_rate']))
    np.save(RESULTS / 'best_q_table.npy', best[2])
    baseline_metrics = next(r for r in rows if r['experiment'] == 'baseline')
    with open(RESULTS / 'metrics.csv', 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    with open(RESULTS / 'summary.json', 'w') as f:
        json.dump({'baseline': baseline_metrics, 'best_setting': best[1],
                   'best_config': best[3], 'best_metrics': best[4],
                   'note': 'Best selected among baseline and individual parameter sweeps; not joint tuning.'}, f, indent=2)
    record_video(best[2])
    print('All results saved under results/ and videos/.')

if __name__ == '__main__':
    main()
