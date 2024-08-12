import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import imageio

# Load environment with proper wrapping
env = gym.make('ALE/Frostbite-v5')
env = Monitor(env)  # Using Monitor from stable_baselines3

# Create DQN agent with changed hyperparameters
model = DQN(
    'CnnPolicy',
    env,
    verbose=1,
    buffer_size=20000,  # Larger buffer size
    learning_starts=1000,
    target_update_interval=1000,  # Longer update interval
    learning_rate=0.00005,  # Lower learning rate
    batch_size=64,  # Bigger batch size
    exploration_fraction=0.1,  # Training period fraction for exploration rate
    exploration_final_eps=0.01,  # Final exploration rate
)

# Train agent and track rewards
rewards = []
episodes = 20  # Number of episodes
timesteps = 10000  # Timesteps per episode

for episode in range(episodes):
    model.learn(total_timesteps=timesteps, log_interval=4)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    rewards.append(mean_reward)
    print(f"Episode {episode+1}/{episodes} - Mean reward: {mean_reward} +/- {std_reward}")

# Save model
model.save("dqn_frostbite_v4")

# Plot rewards over time
plt.plot(rewards, label='Mean Reward')

# Add a moving average trendline
window = 3
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window - 1, len(rewards)), moving_avg, label='Trendline (Moving Avg)', linestyle='--')

plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Training Progress')
plt.legend()

# Save plot to cwd
plt.savefig('training_progress_with_moving_average_v4.png')

# Clear plot for next one
plt.clf()

# Plot rewards with different trendline
plt.plot(rewards, label='Mean Reward')

# Add a moving average trendline
window = 2  # Adjusted window size
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window - 1, len(rewards)), moving_avg, "r--", label='Moving Avg Trendline')

plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Training Progress')
plt.legend()

# Save plot to cwd
plt.savefig('training_progress_with_trendline_v4.png')

# Clear plot for next one
plt.clf()

# Evaluate agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Evaluation (mean +/- std) - Mean reward: {mean_reward} +/- {std_reward}")

# Record video of agent playing
env = gym.make('ALE/Frostbite-v5', render_mode='rgb_array')
env = Monitor(env)  # Using Monitor from stable_baselines3
env = RecordVideo(env, video_folder="video", name_prefix="dqn_frostbite_v4", episode_trigger=lambda x: x == 0)

obs, info = env.reset()
done = False
frames = []

while not done:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    else:
        print("No frame")
    done = terminated or truncated
    print(f"Action: {action}, terminated: {terminated}, truncated: {truncated}")

    if done:
        print(f"Episode finished: terminated={terminated}, truncated={truncated}")
        break

env.close()

# Save video as gif
imageio.mimsave('frostbite_agent_v4.gif', frames, fps=30)
