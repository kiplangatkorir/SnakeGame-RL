import pygame
import random
import numpy as np

class SnakeGame:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (1, 0)
        self.score = 0
        self.food = self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    def _place_food(self):
        x = random.randint(0, (self.width-20)//20) * 20
        y = random.randint(0, (self.height-20)//20) * 20
        return (x, y)

    def _get_state(self):
        # State representation for RL agent
        head = self.snake[0]
        return {
            'snake': self.snake,
            'food': self.food,
            'direction': self.direction,
            'score': self.score
        }

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Place new food or just move
        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(20)
        
        # 6. Return game state
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        # Hits boundary
        if pt[0] < 0 or pt[0] >= self.width or pt[1] < 0 or pt[1] >= self.height:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        
        # Draw snake head (different color and shape)
        head = self.snake[0]
        pygame.draw.circle(self.display, (0, 255, 0), (head[0]+10, head[1]+10), 10)
        
        # Draw eyes on head (facing direction)
        eye_offset_x = 5 if self.direction[0] == 1 else -5 if self.direction[0] == -1 else 0
        eye_offset_y = 5 if self.direction[1] == 1 else -5 if self.direction[1] == -1 else 0
        pygame.draw.circle(self.display, (255, 255, 255), 
                          (head[0]+10+eye_offset_x, head[1]+10+eye_offset_y), 3)
        
        # Draw snake body (connected segments)
        for i in range(1, len(self.snake)):
            prev = self.snake[i-1]
            curr = self.snake[i]
            
            # Draw connecting line between segments
            pygame.draw.line(self.display, (0, 200, 0), 
                            (prev[0]+10, prev[1]+10), 
                            (curr[0]+10, curr[1]+10), 8)
            
            # Draw body joint circles
            pygame.draw.circle(self.display, (0, 200, 0), (curr[0]+10, curr[1]+10), 6)
        
        # Draw food as an apple
        pygame.draw.circle(self.display, (255, 0, 0), (self.food[0]+10, self.food[1]+10), 8)
        pygame.draw.rect(self.display, (0, 150, 0), pygame.Rect(self.food[0]+5, self.food[1]-3, 10, 5))
        
        font = pygame.font.SysFont('arial', 25)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Action: 0=straight, 1=right, 2=left
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)
        
        if action == 1:  # Right turn
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:  # Left turn
            new_dir = clock_wise[(idx - 1) % 4]
        else:  # Straight
            new_dir = self.direction
            
        self.direction = new_dir
        
        x = (self.snake[0][0] + self.direction[0] * 20) % self.width
        y = (self.snake[0][1] + self.direction[1] * 20) % self.height
        self.snake.insert(0, (x, y))
