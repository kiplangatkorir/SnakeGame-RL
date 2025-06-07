from snake_game import SnakeGame
from rl_agent import RLAgent
import numpy as np
import time

def train():
    game = SnakeGame()
    agent = RLAgent(state_size=11, action_size=3)
    batch_size = 1000
    episodes = 1000
    
    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        
        while True:
            # Get action
            action = agent.act(state)
            
            # Perform action
            reward, done, score = game.play_step(action)
            next_state = game._get_state()
            
            # Remember
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e}, Score: {score}, Epsilon: {agent.epsilon:.2f}, Reward: {total_reward}")
                break
            
            # Train
            loss = agent.replay(batch_size)
            
    # Save model
    torch.save(agent.model.state_dict(), 'snake_rl.pth')

if __name__ == "__main__":
    train()
