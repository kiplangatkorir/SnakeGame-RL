from snake_game import SnakeGame
from rl_agent import RLAgent
import numpy as np
import time
import torch

def train():
    game = SnakeGame()
    agent = RLAgent()
    
    episodes = 2000  # More episodes for complex game
    best_score = -float('inf')
    
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
            
            # Train more frequently with smaller batches
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay(agent.batch_size)
            
            if done:
                if score > best_score:
                    best_score = score
                    torch.save(agent.model.state_dict(), 'snake_rl.pth')
                
                print(f"Episode: {e}, Score: {score}, Best: {best_score}, Epsilon: {agent.epsilon:.2f}, Reward: {total_reward}")
                break

if __name__ == "__main__":
    train()
