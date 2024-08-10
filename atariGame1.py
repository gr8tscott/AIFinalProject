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

# Create the DQN agent with adjusted hyperparameters
model = DQN(
    'CnnPolicy', 
    env, 
    verbose=1, 
    buffer_size=20000,  # Increased buffer size
    learning_starts=1000, 
    target_update_interval=1000,  # Increase update interval
    learning_rate=0.00005,  # Adjusted learning rate
    batch_size=64,  # 32Adjusted batch size
    exploration_fraction=0.1,  # Fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps=0.01,  # Final exploration rate
)

# Train the agent and record rewards
rewards = []
num_episodes = 20  # Increase number of training episodes
timesteps_per_episode = 10000  # Increase timesteps per episode

for episode in range(num_episodes):
    model.learn(total_timesteps=timesteps_per_episode, log_interval=4)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    rewards.append(mean_reward)
    print(f"Episode {episode+1}/{num_episodes} - Mean reward: {mean_reward} +/- {std_reward}")

# Save the model
model.save("dqn_frostbite_v4")

# # Plot rewards over time
# plt.plot(rewards)
# plt.xlabel('Training Episode')
# plt.ylabel('Mean Reward')
# plt.title('Training Progress')
# # Save the plot to the current working directory
# plt.savefig('training_progress_v3.png')
# # plt.show()
plt.plot(rewards, label='Mean Reward')

# Add a moving average trendline
window_size = 3
moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size - 1, len(rewards)), moving_average, label='Trendline (Moving Average)', linestyle='--')

plt.xlabel('Training Episode')
plt.ylabel('Mean Reward')
plt.title('Training Progress')
plt.legend()

# Save the plot to the current working directory
plt.savefig('training_progress_with_moving_average_v4.png')

# Clear the plot to avoid overlap if running multiple plots in a script
plt.clf()

# Plot rewards over time################
plt.plot(rewards, label='Mean Reward')

# Add a moving average trendline
window_size = 2  # Adjust the window size for smoothing
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size - 1, len(rewards)), moving_avg, "r--", label='Moving Average Trendline')

plt.xlabel('Training Episode')
plt.ylabel('Mean Reward')
plt.title('Training Progress')
plt.legend()

# Save the plot to the current working directory
plt.savefig('training_progress_with_trendline_v4.png')

# Clear the plot to avoid overlap if running multiple plots in a script
plt.clf()


# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Evaluation Check (mean +/- std) - Mean reward: {mean_reward} +/- {std_reward}")

# Record a video of the agent playing the game
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
        print("Frame is None")
    done = terminated or truncated
    print(f"Action taken: {action}, terminated: {terminated}, truncated: {truncated}")

    if done:
        print(f"Episode finished: terminated={terminated}, truncated={truncated}")
        break

env.close()

# Save video as gif
imageio.mimsave('frostbite_agent_v4.gif', frames, fps=30)
