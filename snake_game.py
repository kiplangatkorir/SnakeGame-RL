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
        
        # Generate obstacles
        self.obstacles = []
        for _ in range(5):
            x = random.randint(0, (self.width-20)//20) * 20
            y = random.randint(0, (self.height-20)//20) * 20
            self.obstacles.append((x, y))
        
        # Food types (normal, bonus, poison)
        self.food_types = ['normal', 'bonus', 'poison']
        self.food_colors = [(255, 0, 0), (255, 215, 0), (138, 43, 226)]

    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (1, 0)
        self.score = 0
        self.food = self._place_food()
        self.food_type = random.choice(self.food_types)
        self.frame_iteration = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width-20)//20) * 20
            y = random.randint(0, (self.height-20)//20) * 20
            pos = (x, y)
            if pos not in self.snake and pos not in self.obstacles:
                return pos

    def _get_state(self):
        # Convert state to numerical array for RL
        head = self.snake[0]
        food = self.food
        
        # Calculate danger directions with obstacles
        danger_straight, danger_right, danger_left = self._check_dangers(head)
        
        # Relative food position
        food_left = food[0] < head[0]
        food_right = food[0] > head[0]
        food_up = food[1] < head[1]
        food_down = food[1] > head[1]
        
        # Current direction
        dir_left = self.direction[0] == -1
        dir_right = self.direction[0] == 1
        dir_up = self.direction[1] == -1
        dir_down = self.direction[1] == 1
        
        # Create state array
        state = [
            # Danger directions
            danger_straight,
            danger_right,
            danger_left,
            
            # Current movement direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # Food location
            food_left,
            food_right,
            food_up,
            food_down,
            
            # Food type (one-hot encoded)
            self.food_type == 'normal',
            self.food_type == 'bonus',
            self.food_type == 'poison'
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _check_dangers(self, head):
        # Check dangers in all directions
        directions = [
            self.direction,  # straight
            (self.direction[1], -self.direction[0]),  # right
            (-self.direction[1], self.direction[0])   # left
        ]
        
        dangers = []
        for dir in directions:
            new_pos = (head[0] + dir[0]*20, head[1] + dir[1]*20)
            dangers.append(
                self._is_collision(new_pos) or 
                new_pos in self.obstacles
            )
        
        return dangers

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
        
        # 4. Handle food collision
        if self.snake[0] == self.food:
            if self.food_type == 'normal':
                self.score += 1
                reward = 10
            elif self.food_type == 'bonus':
                self.score += 3
                reward = 20
            else:  # poison
                self.score -= 2
                reward = -20
                if len(self.snake) > 3:
                    self.snake.pop()
                    self.snake.pop()
            
            self.food = self._place_food()
            self.food_type = random.choice(self.food_types)
        else:
            self.snake.pop()
        
        # 5. Update UI
        self._update_ui()
        self.clock.tick(20)
        
        # 6. Return game state
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(obs[0], obs[1], 20, 20))
        
        # Draw snake head
        head = self.snake[0]
        pygame.draw.circle(self.display, (0, 255, 0), (head[0]+10, head[1]+10), 10)
        
        # Draw eyes
        eye_offset_x = 5 if self.direction[0] == 1 else -5 if self.direction[0] == -1 else 0
        eye_offset_y = 5 if self.direction[1] == 1 else -5 if self.direction[1] == -1 else 0
        pygame.draw.circle(self.display, (255, 255, 255), (head[0]+10+eye_offset_x, head[1]+10+eye_offset_y), 3)
        
        # Draw snake body
        for i in range(1, len(self.snake)):
            prev = self.snake[i-1]
            curr = self.snake[i]
            pygame.draw.line(self.display, (0, 200, 0), (prev[0]+10, prev[1]+10), (curr[0]+10, curr[1]+10), 8)
            pygame.draw.circle(self.display, (0, 200, 0), (curr[0]+10, curr[1]+10), 6)
        
        # Draw food based on type
        food_color = self.food_colors[self.food_types.index(self.food_type)]
        pygame.draw.circle(self.display, food_color, (self.food[0]+10, self.food[1]+10), 8)
        
        # Score display
        font = pygame.font.SysFont('arial', 25)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

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
