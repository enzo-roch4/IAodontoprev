import cv2
import requests
import numpy as np
from roboflow import Roboflow

# Configuração da API Roboflow
api_key = "API_KEY"  # lugar onde ficara a nossa api responsavel pela analise
workspace_name = "workspace"  # Nome do seu workspace no Roboflow
project_name = "IAodontoprev"  # Nome do projeto no Roboflow
version = "1" 

rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace_name).project(project_name)
model = project.version(version).model

def analyze_image(image_path):
    # Carregar imagem
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Fazer a previsão com o modelo
    prediction = model.predict(image_path, confidence=40, overlap=30).json()
    
    # Analisar os resultados
    for result in prediction["predictions"]:
        x, y, width, height = result["x"], result["y"], result["width"], result["height"]
        class_name = result["class"]
        confidence = result["confidence"]

        # Desenhar o retângulo na imagem
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        color = (255, 0, 0)  # Azul
        cv2.rectangle(image, start_point, end_point, color, 2)
        cv2.putText(image, f"{class_name} ({confidence:.2f})", (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar a imagem com previsões
    cv2.imshow("Resultado da Análise", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Teste com uma imagem de exemplo
analyze_image("caminho/para/sua/imagem.jpg")
