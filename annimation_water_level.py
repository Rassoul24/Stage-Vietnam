import pygame
import sys
import numpy as np
import math

def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=10, space_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos

    # Calculer la distance et l'angle entre les points
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)

    # Nombre total de dash+space
    dash_space_length = dash_length + space_length
    num_dashes = int(length // dash_space_length)

    for i in range(num_dashes + 1):
        start_x = x1 + (i * dash_space_length) * math.cos(angle)
        start_y = y1 + (i * dash_space_length) * math.sin(angle)

        end_x = start_x + dash_length * math.cos(angle)
        end_y = start_y + dash_length * math.sin(angle)

        # Ne pas dépasser la fin de la ligne
        if math.hypot(end_x - x1, end_y - y1) > length:
            end_x = x2
            end_y = y2

        pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)

def animation(y_true, y_pred,dates):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    screen_width, screen_height = screen.get_size()
    pygame.display.set_caption("Animation des hauteurs avec vagues")

    # Couleurs
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE_1 = (0, 100, 255)
    BLUE_2 = (0, 50, 180)

    Surestimation = []
    Sous_estimation = []
    Bonne_estimation = []

    # Initialisation
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    running = True
    temps = 0
    text_rect_1 = pygame.Rect(screen_width//5, 50, 300, 3)
    text_rect_2 = pygame.Rect(3*screen_width//5, 50, 300, 3)
    text_rect_3 = pygame.Rect(screen_width//5, 100, 300, 30)
   

    while running:
        clock.tick(7)
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if temps >= len(y_true):
            break

        y1 = 300 - y_true[temps] * 10
        y2 = 300 - y_pred[temps] * 10

        # Texte
        text_1 = font.render(f"Niveau réel = {y_true[temps]:.2f}", True, RED)
        text_2 = font.render(f"Niveau prédit = {y_pred[temps]:.2f}", True, BLUE_2)
        text_3 = font.render(f"Date = {dates[temps]}", True, BLUE_2)

        screen.blit(text_1, text_rect_1)
        screen.blit(text_2, text_rect_2)
        screen.blit(text_3,text_rect_3)

        # Comparaison
        if y1 < y2:
            Surestimation.append(1)
        elif y1 > y2:
            Sous_estimation.append(1)
        else:
            Bonne_estimation.append(1)

        # Dessin des vagues
        def draw_wave(y_base,x_base,color, phase):
            points = []
            for x in range(x_base, x_base+screen_width//2-0.1, 4):
                y = y_base + 10 * math.sin((x * 0.05) + phase)
                points.append((x, y))
            
            points.append((x_base+screen_width//2, screen_height))
            points.append((x_base, screen_height))
            #points.append((x_base, screen_height))
            pygame.draw.polygon(screen, color, points)
        

        draw_wave(y1,0 ,BLUE_1, temps)
        draw_wave(y2,screen_width//2, BLUE_1, temps+0.5 )
        y_moyenne = 300 - np.mean(y_true) * 10  # ajuster pour coord. écran
        pygame.draw.line(screen, RED, (0, y_moyenne), (screen_width, y_moyenne),5)

        #pygame.draw.line(screen, RED, (screen_width//2, 0), (screen_width//2, screen_height), 3)  # épaisseur 3 pixels
        #draw_dashed_line(screen, (255, 0, 0), (screen_width//2, 0), (screen_width//2, screen_height), width=2, dash_length=15, space_length=10)


        pygame.display.flip()
        temps += 1

    # Statistiques
    print(f"Proba Surestimation = {sum(Surestimation)/len(y_true):.2f}")
    print(f"Proba Sousestimation = {sum(Sous_estimation)/len(y_pred):.2f}")
    print(f"Bonne estimation     = {sum(Bonne_estimation)/len(y_true):.2f}")

    pygame.quit()
    sys.exit()

# Exemple
if __name__ == "__main__":
    y1 = np.random.randint(5, 10, size=365)
    y2 = np.random.randint(5, 10, size=365)
    animation(y1, y2)
