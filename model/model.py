import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Usar backend no interactivo
import matplotlib.pyplot as plt

class EyeTrackerCalibration:
    def __init__(self, screen_points, eye_tracker_data):
        # Transforma a np.array las listas
        self.raw_screen_points = np.array(screen_points, dtype=np.float32)
        self.raw_eye_tracker_data = np.array(eye_tracker_data, dtype=np.float32)
        # Escalar los puntos al mismo rango antes de calcular la homografía
        self.screen_points, self.screen_scale, self.screen_offset = self.scale_data(self.raw_screen_points)
        self.eye_tracker_data, self.eye_scale, self.eye_offset = self.scale_data(self.raw_eye_tracker_data)
        # Define la matriz de homografia
        self.H = None 
    
    def scale_data(self, points):
        # Escala los puntos a un rango [0, 1] usando normalización min-max
        min_val = np.min(points, axis=0)
        max_val = np.max(points, axis=0)
        scale = max_val - min_val
        scaled_points = (points - min_val) / scale
        return scaled_points, scale, min_val

    def unscale_point(self, point):
        # Convierte un punto de la escala normalizada a la escala de la pantalla original
        return (point * self.screen_scale) + self.screen_offset

    def compute_homography(self):
        # Calcula la matriz de homografía
        self.H, _ = cv2.findHomography(self.eye_tracker_data, self.screen_points, cv2.RANSAC)

    def transform_gaze_point(self, gaze_point):
        # Transforma un punto detectado por el eye tracker a coordenadas de la pantalla
        if self.H is None:
            raise ValueError("La homografía no ha sido calculada. Llama a compute_homography() primero.")
        
        # Escalar el punto de entrada
        normalized_gaze = (np.array(gaze_point, dtype=np.float32) - self.eye_offset) / self.eye_scale
        # Transformar el punto normalizado con la homografía
        point = np.array([[normalized_gaze[0], normalized_gaze[1], 1]], dtype=np.float32).T
        transformed_point = self.H @ point  # Multiplicación de matrices
        transformed_point /= transformed_point[2]  # Normalización

        # Desescalar el resultado para volver a la escala original de la pantalla
        final_point = self.unscale_point(transformed_point[:2].flatten())

        # Asegurar que los valores están dentro del rango de la pantalla
        final_point[0] = np.clip(final_point[0], np.min(self.raw_screen_points[:, 0]), np.max(self.raw_screen_points[:, 0]))
        final_point[1] = np.clip(final_point[1], np.min(self.raw_screen_points[:, 1]), np.max(self.raw_screen_points[:, 1]))

        return final_point

    def visualize_calibration(self, filename="calibration_result_homography_scaled.png"):
        # Genera un gráfico mostrando los puntos antes y después de la calibración y lo guarda en un archivo
        plt.figure(figsize=(10, 8))

        # Puntos antes de la transformación (sin calibrar)
        plt.scatter(self.raw_eye_tracker_data[:, 0], self.raw_eye_tracker_data[:, 1], 
                    color='red', label="Datos Eye Tracker (sin calibrar)", marker='x')

        # Puntos después de la calibración (en la pantalla)
        transformed_points = np.array([self.transform_gaze_point(p) for p in self.raw_eye_tracker_data])
        plt.scatter(transformed_points[:, 0], transformed_points[:, 1], 
                    color='blue', label="Datos Calibrados (pantalla)", marker='o')

        # Puntos de referencia en la pantalla
        plt.scatter(self.raw_screen_points[:, 0], self.raw_screen_points[:, 1], 
                    color='green', label="Puntos de Referencia", marker='s')

        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.legend()
        plt.title("Comparación de Puntos Antes y Después de la Calibración (Homografía Escalada)")
        plt.grid(True)

        # Guardar la imagen en un archivo en lugar de mostrarla
        plt.savefig(filename)
        print(f"Gráfico guardado en {filename}")

# Puntos de referencia en la pantalla (X', Y')
data_estimulo = [(137, 105),(620, 88),(1162, 70),(95, 345),(616, 340),(1190, 344),(51, 618),(605, 634),(1226, 652)]
# Puntos detectados por el eye tracker (X, Y)
data_eye = [(242, 264),(195, 260),(145, 253),(245, 290),(194, 278),(156, 258),(273, 280),(201, 295),(142, 267)]

# Crear objeto de calibración con homografía escalada
calibration = EyeTrackerCalibration(data_estimulo, data_eye)
calibration.compute_homography()

# Probar transformación de un nuevo punto
gaze_point = [520, 490]  # Coordenadas detectadas por el eye tracker
corrected_point = calibration.transform_gaze_point(gaze_point)

# Mostrar resultado corregido en la terminal
print("Punto corregido en la pantalla:", corrected_point)

# Generar y guardar la imagen de la calibración
calibration.visualize_calibration("calibration_result_homography_scaled.png")
