import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import imageio

# Create env with video recording
env = gym.make('ALE/Frostbite-v5', render_mode='rgb_array')
env = RecordVideo(env, video_folder="video", name_prefix="dqn_frostbite_v5", episode_trigger=lambda x: x == 0)

# Load pre-trained model
model = DQN.load("dqn_frostbite_v3", env=env)

# Evaluate the model
mean_rwd, std_rwd = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Final eval - Mean rwd: {mean_rwd} +/- {std_rwd}")

# Prepare env for recording video

# Run agent and record
obs, info = env.reset()
done = False
frames = []

while not done:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render frame and append to list
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    done = terminated or truncated

env.close()  # Close env properly

# Save as GIF
if frames:
    imageio.mimsave('frostbite_agent_v5.gif', frames, fps=30)
else:
    print("No frames. GIF creation failed.")
