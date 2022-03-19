import cv2
from matplotlib import pyplot as plt
import videoCapture as video

# Como parâmetro também pode ser o diretório do video
captura = cv2.VideoCapture(0)

if __name__ == "__main__":
  
  # Imagem para ser detectada
  toDetect = cv2.imread("image/toDetect.jpeg")
  
  # Selecionar zona de interesse
  box = cv2.selectROI("select roi", toDetect, fromCenter=False, showCrosshair=False)
  print(box)

  # Tracker responsável por detectar os objetos
  tracker = cv2.TrackerCSRT_create()
  
  tracker.init(toDetect, box)

  while captura.isOpened():
    
    ret, frame = captura.read()
    
    if not ret:
      break
    
    ok, box  = tracker.update(frame)
    
    # Criação do retângulo mostrado
    if ok:
      print('ok')
      pt1 = (box[0], box[1])
      pt2 = ((box[0]+box[2]), (box[1] + box[3]))
      cv2.rectangle(frame, pt1, pt2, (255,0,0),2,1)
      
    else:
      print('FALHA')
    
    # Mostrar o video  
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) == ord('q'):
      break 
    
  # Retirar vestígios na memória  
  captura.release()
  cv2.destroyAllWindows()
      