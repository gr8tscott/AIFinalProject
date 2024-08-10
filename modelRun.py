import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import imageio


env = gym.make('ALE/Frostbite-v5', render_mode='rgb_array')
env = RecordVideo(env, video_folder="video", name_prefix="dqn_frostbite_v5", episode_trigger=lambda x: x == 0)

# Load the pre-trained model
model = DQN.load("dqn_frostbite_v3", env=env)

# Reset the environment and initialize variables
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Final evaluation - Mean reward: {mean_reward} +/- {std_reward}")

# Set up the environment for recording the final video

# Run the agent in the environment and record the video
obs, info = env.reset()
done = False
frames = []

while not done:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the frame and add it to the list
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    done = terminated or truncated

env.close()  # Ensure environment is properly closed

# Save the frames as a GIF
if frames:
    imageio.mimsave('frostbite_agent_v5.gif', frames, fps=30)
else:
    print("No frames collected. GIF creation failed.")