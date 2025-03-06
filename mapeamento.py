import cv2
from ultralytics import YOLO
import threading
import time
import requests

# Variável global para indicar se há pessoa (1) ou não (0)
person = 0
# Mutex para sincronizar o acesso à variável global
lock = threading.Lock()
# Carregar o modelo YOLOv8
model = YOLO("yolov8n.pt")
# Carregar vídeo em vez da webcam
video_path = "testeEduardo.mp4"  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)
# Definir nova largura e altura desejadas
new_width = 600
new_height = 800

# URL que desejamos acessar
url_liga = "http://192.168.1.11/H"
url_desliga = "http://192.168.1.11/L"

def monitor_person():
    global person
    while True:
        with lock:
            if person > 0:
                print("AREA DE RISCO")
                response = requests.get(url_liga)
                # Verifica se a requisição foi bem sucedida (código 200)
                if response.status_code == 200:
                    print("ALARME LIGADO!")
                else:
                    print("ALARME NÃO ENCONTRADO!")
            else:
                print("AREA DE LIVRE")
                response = requests.get(url_desliga)
                # Verifica se a requisição foi bem sucedida (código 200)
                if response.status_code == 200:
                    print("ALARME DELIGADO!")
                else:
                    print("ALARME NÃO ENCONTRADO!")
        time.sleep(1)

# Inicia a thread de monitoramento antes do loop principal
monitor_thread = threading.Thread(target=monitor_person)
monitor_thread.start()

while True:
    success, frame = cap.read()
    if not success:
        break
    resized_frame = cv2.resize(frame, (new_width, new_height))
    results = model.predict(resized_frame, verbose=False, classes=0, conf=0.8)
    annotated_frame = results[0].plot()
    # Obter coordenadas das bounding boxes
    for r in results:
        for box in r.boxes:
            x, y, w, h = map(int, box.xywh[0])  # Coordenadas (x, y, largura, altura)
            point_ref = y + h
            #print(f"Objeto detectado em X: {x}, Y: {point_ref}, Largura: {w}, Altura: {h}")
            if (point_ref > 850):
                with lock:
                    person = 1
            else:
                with lock:
                    person = 0    
    cv2.imshow("Detecção de Pessoas", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()