import pygame
import numpy as np
import random
import os

# Initialisation de Pygame
pygame.init()

# Paramètres du jeu
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20

display = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Couleurs
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

test_count = 1  # Compteur de tests
best_score = 0  # Stocker le meilleur score atteint
score = 0  # Score en cours

# Actions possibles (Haut, Bas, Gauche, Droite)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Q-Learning Parameters
learning_rate = 0.5  # Réduction pour stabiliser l’apprentissage
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.997  # Ne descend pas trop vite

def load_q_table():
    """Charge la Q-Table si elle existe."""
    global q_table, test_count, best_score
    if os.path.exists("q_table.npy"):
        q_table = np.load("q_table.npy")
        print("Q-Table chargée avec succès !")
    else:
        q_table = np.zeros((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE, len(actions)))
        print("Aucune Q-Table trouvée, démarrage avec une table vide.")
    
    if os.path.exists("test_count.npy"):
        test_count = int(np.load("test_count.npy"))  # Convertir en int
    if os.path.exists("best_score.npy"):
        best_score = int(np.load("best_score.npy")) 

def save_q_table():
    """Sauvegarde la Q-Table, le compteur de tests et le score."""
    np.save("q_table.npy", q_table)
    np.save("test_count.npy", test_count)
    np.save("best_score.npy", best_score)
    print("Q-Table et compteur de tests sauvegardés !")

load_q_table()

def get_state(snake, food):
    """Retourne la position du serpent sous forme d’état"""
    max_x = WIDTH // GRID_SIZE - 1
    max_y = HEIGHT // GRID_SIZE - 1
    
    state_x = min(max(snake[0][0] // GRID_SIZE, 0), max_x)
    state_y = min(max(snake[0][1] // GRID_SIZE, 0), max_y)

    return (state_x, state_y)

def choose_action(state):
    """Choisit une action selon la Q-Table (exploration vs exploitation)"""
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(range(len(actions)))
    else:
        return np.argmax(q_table[state[0], state[1]])

def update_q_table(state, action, reward, next_state):
    """Met à jour la Q-Table avec la règle du Q-Learning"""
    max_x = WIDTH // GRID_SIZE - 1
    max_y = HEIGHT // GRID_SIZE - 1

    next_x = min(max(next_state[0], 0), max_x)
    next_y = min(max(next_state[1], 0), max_y)

    best_next_action = np.max(q_table[next_x, next_y])  
    q_table[state[0], state[1], action] += learning_rate * (reward + discount_factor * best_next_action - q_table[state[0], state[1], action])

def reset_game():
    """Réinitialise le jeu et met à jour le compteur"""
    global test_count, best_score, score
    if score > best_score:
        best_score = score  
    test_count += 1  
    score = 0  
    return [(WIDTH // 2, HEIGHT // 2)], generate_food([])

def generate_food(snake):
    """Génère une pomme à une position non occupée par le serpent"""
    while True:
        food = (random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE, 
                random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE)
        if food not in snake:
            return food

snake, food = reset_game()
direction = (0, -1)
running = True

while running:
    display.fill(BLACK)
    pygame.draw.rect(display, RED, (*food, GRID_SIZE, GRID_SIZE))
    for segment in snake:
        pygame.draw.rect(display, GREEN, (*segment, GRID_SIZE, GRID_SIZE))
    
    # Affichage du score, meilleur score et numéro du test
    font = pygame.font.SysFont(None, 30)
    score_text = font.render(f"Test #{test_count} | Score: {score} | Best: {best_score}", True, WHITE)
    display.blit(score_text, (10, 10))
    
    pygame.display.update()
    clock.tick(5000)  # Entraînement ultra rapide

    exploration_rate = max(0.01, exploration_rate * exploration_decay)

    state = get_state(snake, food)
    action = choose_action(state)
    direction = actions[action]

    new_head = (snake[0][0] + direction[0] * GRID_SIZE, snake[0][1] + direction[1] * GRID_SIZE)

    # Vérification des limites et passage à travers les murs
    new_head = (new_head[0] % WIDTH, new_head[1] % HEIGHT)

    # Calcul de la récompense en fonction du déplacement
    old_distance = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])
    new_distance = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])

    reward = -0.1  # Pénalité légère par défaut

    if new_distance < old_distance:
        reward += 10   # Bonus si on s’approche
    else:
        reward -= 2  # On diminue la punition pour éviter qu'il bloque

    if new_head == food:
        reward = 500  # Augmentation de la récompense
        score += 1
        snake.insert(0, new_head)
        food = generate_food(snake)
    else:
        snake.insert(0, new_head)
        snake.pop()

    if new_head in snake[1:]:  
        reward = -1000  
        snake, food = reset_game()

    next_state = get_state(snake, food)
    update_q_table(state, action, reward, next_state)

    if test_count % 50 == 0:
        save_q_table()

pygame.quit()
