from cameras import *
import pygame
import sys

class CalibrationApp:
    def __init__(self):
        # Inicializar pygame
        pygame.init()

        # Configurar colores y fuente
        self.BACKGROUND_COLOR = (30, 30, 30)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.FONT_COLOR = (200, 200, 200)

        # Configurar pantalla
        self.screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
        self.WIDTH, self.HEIGHT = self.screen.get_size()
        pygame.display.set_caption("Eye Tracking Calibration")

        self.font = pygame.font.SysFont(None, 36)

        # Definir margen para acercar los puntos a los bordes
        self.MARGIN_X = 100
        self.MARGIN_Y = 100

        # Espaciado ajustado para que los puntos cubran mejor la pantalla
        self.spacing_x = (self.WIDTH - 2 * self.MARGIN_X) // 2
        self.spacing_y = (self.HEIGHT - 2 * self.MARGIN_Y) // 2

        # Coordenadas de los puntos de calibración (3x3)
        self.points = [
            (self.MARGIN_X + j * self.spacing_x, self.MARGIN_Y + i * self.spacing_y)
            for i in range(3) for j in range(3)
        ]

        # Crear botones
        self.calibrate_button = pygame.Rect((self.WIDTH // 2) - 250, self.HEIGHT - 80, 150, 40)
        self.recalibrate_button = pygame.Rect((self.WIDTH // 2) + 100, self.HEIGHT - 80, 150, 40)

        # Estado de calibración
        self.current_point_index = 0

        # Reloj para controlar FPS
        self.clock = pygame.time.Clock()

    def draw_screen(self):
        """Dibuja la pantalla con los puntos y botones."""
        self.screen.fill(self.BACKGROUND_COLOR)

        # Dibujar puntos (el punto activo se muestra en rojo)
        for i, point in enumerate(self.points):
            if i == self.current_point_index:
                pygame.draw.circle(self.screen, self.RED, point, 30)
            else:
                pygame.draw.circle(self.screen, self.WHITE, point, 30)

        # Dibujar botones
        pygame.draw.rect(self.screen, self.WHITE, self.calibrate_button, 2)
        pygame.draw.rect(self.screen, self.WHITE, self.recalibrate_button, 2)

        # Texto en botones
        text = self.font.render('Calibrate', True, self.WHITE)
        self.screen.blit(text, (self.calibrate_button.x + 15, self.calibrate_button.y + 5))
        text = self.font.render('Re-Calibrate', True, self.WHITE)
        self.screen.blit(text, (self.recalibrate_button.x + 5, self.recalibrate_button.y + 5))

    def start_calibration(self):
        """Avanza al siguiente punto de calibración."""
        if self.current_point_index < len(self.points) - 1:
            self.current_point_index += 1

    def reset_calibration(self):
        """Reinicia la calibración al primer punto."""
        self.current_point_index = 0

    def run(self):
        """Bucle principal de la aplicación."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Manejar clics de mouse en botones
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if self.calibrate_button.collidepoint(mouse_pos):
                        self.start_calibration()
                    if self.recalibrate_button.collidepoint(mouse_pos):
                        self.reset_calibration()

            # Actualizar pantalla
            self.draw_screen()
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == '__main__':
    app = CalibrationApp()
    app.run()
