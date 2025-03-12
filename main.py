from cameras import *
from paterns import *
import cv2 
pupil_detector = PupilDetector()
world_camera = WorldCamera()
#calibration = CalibrationApp()


while True:
    ret_eye, eye_frame = pupil_detector.cap.read()
    ret_world, world_frame = world_camera.cap.read()
    
    if not ret_eye or not ret_world:
        print("Error: No se pudo capturar el fotograma de una de las cámaras.")
        break
    
    #Procesar los frames
    eye_frame, pupil_data = pupil_detector.process_frame(eye_frame)
    #world_frame = world_camera.process_frame(world_frame, pupil_data)
    
    #Mostrar los frames
    world_camera.show_frame(world_frame)
    pupil_detector.show_frame(eye_frame)
    #calibration.run()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pupil_detector.release_resources()
world_camera.release_resources()
cv2.destroyAllWindows()
