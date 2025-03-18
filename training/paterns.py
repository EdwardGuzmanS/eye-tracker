# training/paterns.py
import pygame
import sys
import queue
from collections import deque
from common.cameras import *

class CalibrationApp:
    def __init__(self):
        pygame.init()
        self.BACKGROUND_COLOR = (30, 30, 30)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.FONT_COLOR = (200, 200, 200)
        self.screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
        self.WIDTH, self.HEIGHT = self.screen.get_size()
        pygame.display.set_caption("Calibración de Eye Tracking")
        self.font = pygame.font.SysFont(None, 36)
        self.MARGIN_X = 100
        self.MARGIN_Y = 100
        self.spacing_x = (self.WIDTH - 2 * self.MARGIN_X) // 2
        self.spacing_y = (self.HEIGHT - 2 * self.MARGIN_Y) // 2
        self.points = [
            (self.MARGIN_X + j * self.spacing_x, self.MARGIN_Y + i * self.spacing_y)
            for i in range(3) for j in range(3)
        ]
        self.calibrate_button = pygame.Rect((self.WIDTH // 2) - 250, self.HEIGHT - 80, 150, 40)
        self.recalibrate_button = pygame.Rect((self.WIDTH // 2) + 100, self.HEIGHT - 80, 150, 40)
        self.current_point_index = 0
        self.clock = pygame.time.Clock()
    
    def draw_screen(self):
        self.screen.fill(self.BACKGROUND_COLOR)
        for i, point in enumerate(self.points):
            color = self.RED if i == self.current_point_index else self.WHITE
            pygame.draw.circle(self.screen, color, point, 30)
        pygame.draw.rect(self.screen, self.WHITE, self.calibrate_button, 2)
        pygame.draw.rect(self.screen, self.WHITE, self.recalibrate_button, 2)
        text = self.font.render('Calibrate', True, self.WHITE)
        self.screen.blit(text, (self.calibrate_button.x + 15, self.calibrate_button.y + 5))
        text = self.font.render('Re-Calibrate', True, self.WHITE)
        self.screen.blit(text, (self.recalibrate_button.x + 5, self.recalibrate_button.y + 5))
    
    def start_calibration(self):
        if self.current_point_index < len(self.points) - 1:
            self.current_point_index += 1
    
    def reset_calibration(self):
        self.current_point_index = 0

class IntegratedCalibrationApp(CalibrationApp):
    def __init__(self, pupil_queue, shared_calibration_data):
        super().__init__()
        self.calibration_data = deque()
        self.pupil_queue = pupil_queue  # Cola para recibir datos de la pupila
        self.shared_calibration_data = shared_calibration_data  # Lista compartida (Manager.list())
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if self.calibrate_button.collidepoint(mouse_pos):
                        try:
                            pupil_coord = self.pupil_queue.get_nowait()
                            print("Guardando datos de la pupila:", pupil_coord)
                            self.calibration_data.append(pupil_coord)
                        except queue.Empty:
                            print("No se detectaron coordenadas de la pupila.")
                        self.start_calibration()
                    elif self.recalibrate_button.collidepoint(mouse_pos):
                        self.reset_calibration()
                        self.calibration_data.clear()
            self.draw_screen()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        print("Datos de calibración (pupila) capturados:")
        print(self.calibration_data)
        self.shared_calibration_data.extend(self.calibration_data)

def run_calibration(pupil_queue, shared_calibration_data):
    app = IntegratedCalibrationApp(pupil_queue, shared_calibration_data)
    app.run()

if __name__ == '__main__':
    import queue
    q = queue.Queue(maxsize=1)
    shared_data = []  # Prueba local; en producción se usará Manager.list()
    run_calibration(q, shared_data)
