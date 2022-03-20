import cv2
import time
import os

COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

# Classificação de nomes
class_names = []

""" cur_path = os.path.dirname(__file__)
path = os.path.relpath('Redes_Neurais/coco.names', cur_path) """

outfile = open("coco.names", "r")

with open(outfile, "r") as names:
    class_names = [cocoName.strip() for cocoName in names.readlines()]

# Captura de video
captura = cv2.VideoCapture(0)

#Captura de pesos em rede neural
neural = cv2.dnn.readNet("yolov4.tiny.weights", "yolov4-tiny.cfg")

# Parâmetros de rede neural
model = cv2.dnn_ClassificationModel(neural)
model.setInputParams(size=(416,416), scale=1/255)

while True:
    
    # Captura do video
    conected, frame = captura.read()
    
    # Contagem de milisegundos de início
    start = time.time()
    
    # Nível de detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.1)
    
    # Contagem de milisegundos de fim
    end = time.time()
    
    # Percorrer detecções
    for (classId, score, box) in zip(classes, scores, boxes):
        
        cor = COLORS[int(classId) % len(COLORS)]
        
        # Extraindo informações e colocando em string
        textoObjeto = f"{class_names[classId[0]]} : {score}"
        
        # Aplicando a box nas detecções
        cv2.rectangle(frame, box, cor, 2)
        
        #Aplicando as informações da string de detecção
        cv2.putText(frame, textoObjeto, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor)

    # Média de fps
    fps_label = f"FPS: {round((1.0/(end - start)),0)}"
    
    # Escrevendo FPS na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Mostrando a Imagem
    cv2.imshow('Detecção', frame)
    
    # Comando de parar a execução
    if cv2.waitKey(1) == ord('q'):
      break
  
 # Retirar vestígios na memória e fechar cameras
captura.release()
cv2.destroyAllWindows()
