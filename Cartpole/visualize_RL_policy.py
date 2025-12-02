import gymnasium as gym
from gymnasium.envs.registration import register, make as gym_make
from stable_baselines3 import PPO
import argparse
import os

# 1. Environment Registration
# This must match the registration in RL_DDT.py, ensuring the environment is known to gymnasium.
register(
    id="FH-CartPole",
    entry_point="Fixed_Horizon_CP_Env:FH_CartPoleEnv",
    vector_entry_point="Fixed_Horizon_CP_Env:FH_CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=200,
)

def visualize_trained_policy(model_name, soft_routing_argmax, rl_seed):
    """
    Loads a trained PPO model and runs it in the FH-CartPole environment with rendering.
    """
    
    # --- Configuration ---
    # NOTE: These directories must match the saving structure in RL_DDT.py
    dir = './logic/Reward_Models_3/DDT'
    save_model_dir = dir + f'/RL_Models/{model_name}/'
    save_rl_model_dir = save_model_dir + "/model/"
    
    # Stable-Baselines3 PPO models are typically saved as .zip files
    RL_MODEL_PATH = (
        save_rl_model_dir
        + f"PPO_Cartpole_RL_using_{model_name}_softrouting{soft_routing_argmax}_seed{rl_seed}.zip"
    )

    print(f"Attempting to load model from: {RL_MODEL_PATH}")

    if not os.path.exists(RL_MODEL_PATH):
        print("ERROR: Model file not found!")
        print("Please ensure you have:")
        print("1. Run RL_DDT.py with the specified arguments to save the model.")
        print("2. The path to the saved model is correct (check the directory structure).")
        print(f"Expected model name: PPO_Cartpole_RL_using_{model_name}_softrouting{soft_routing_argmax}_seed{rl_seed}.zip")
        return

    # 2. Create a single environment for visualization
    try:
        # Use render_mode="human" for real-time display. Requires pygame.
        eval_env = gym_make("FH-CartPole", render_mode="human")
    except Exception as e:
        print(f"ERROR: Could not create rendering environment. Ensure 'Fixed_Horizon_CP_Env.py' is accessible and libraries (like pygame) are installed. Error: {e}")
        return

    # 3. Load the trained PPO model
    try:
        # PPO loads the policy weights and configuration
        # Pass the environment to the load function, as required by SB3
        visual_model = PPO.load(RL_MODEL_PATH, env=eval_env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load PPO model. Error: {e}")
        eval_env.close()
        return

    # 4. Run the trained policy for evaluation
    num_eval_episodes = 5
    max_steps_per_episode = 200 # Max steps set in environment registration

    for episode in range(num_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Loop until the episode terminates or is truncated
        while not done and not truncated and steps < max_steps_per_episode:
            # Predict the action (deterministic=True is standard for evaluation)
            action, _ = visual_model.predict(obs, deterministic=True)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Render the frame (this displays the window)
            eval_env.render()
            
            total_reward += reward
            steps += 1
            
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Steps = {steps}, Total Reward = {total_reward}")

    # 5. Cleanup
    eval_env.close()
    print("Visualization finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained PPO policy for the FH-CartPole environment.")
    
    # These arguments mirror the ones in RL_DDT.py to locate the correct saved model
    parser.add_argument('--soft_routing_argmax', type=int, default=1, 
                        help="The value (0 or 1) used for soft_routing_argmax during training.")
    parser.add_argument('--RL_seed', type=int, default=0, 
                        help="The RL/PPO seed used during training.")
    parser.add_argument('--model_name', default="BEST_RSS_hard", 
                        help="The model name (e.g., 'BEST_RSS_hard') used during training.")
    
    args = parser.parse_args()
    
    visualize_trained_policy(
        model_name=args.model_name,
        soft_routing_argmax=args.soft_routing_argmax,
        rl_seed=args.RL_seed
    )