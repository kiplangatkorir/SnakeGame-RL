from snake_game import SnakeGame
from rl_agent import RLAgent
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

def train():
    game = SnakeGame()
    agent = RLAgent()
    
    episodes = 2000  # More episodes for complex game
    best_score = -float('inf')
    
    # Track metrics
    scores = []
    avg_scores = []
    epsilons = []
    
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
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
                
                # Update metrics
                scores.append(score)
                avg_scores.append(np.mean(scores[-100:]))  # 100-episode average
                epsilons.append(agent.epsilon)
                
                # Update plots
                ax1.clear()
                ax1.plot(scores, 'b', label='Score')
                ax1.plot(avg_scores, 'r', label='Avg Score (100)')
                ax1.axhline(best_score, color='g', linestyle='--', label='Best Score')
                ax1.set_ylabel('Score')
                ax1.legend()
                
                ax2.clear()
                ax2.plot(epsilons, 'm', label='Epsilon')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Epsilon')
                ax2.legend()
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
                
                print(f"Episode: {e}, Score: {score}, Best: {best_score}, Epsilon: {agent.epsilon:.2f}, Reward: {total_reward}")
                break
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
