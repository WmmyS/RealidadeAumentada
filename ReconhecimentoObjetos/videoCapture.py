from ast import While
import cv2

# Capturar a imagem de dispositivo
def capturarVideo():
    video = cv2.VideoCapture(0);
    
    while True:
        conectado, frame = video.read()
        
        # Exibir video capturado
        cv2.imshow('Video', frame)
        
        # Aguardar a digitação da tecla 'Q'
        if cv2.waitKey(1) == ord('q'):
            break
        
    # Retirar vestígios na memória
    video.release()
    cv2.destroyAllWindows()

        
    