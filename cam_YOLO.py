#!/usr/bin/env python3

"""
Sistema de Detecção com YOLO v8 + Iriun Webcam
- Detecção de pessoas e objetos (YOLO v8)
"""

import cv2
import sys
from datetime import datetime
from ultralytics import YOLO

# ==================== CONFIGURAÇÕES ====================
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5  # Confiança mínima para detecção YOLO
SHOW_FPS = True             # Mostrar FPS na tela

# ==================== CARREGAMENTO DO YOLO ====================
print("=" * 60)
print("Carregando modelo YOLO v8...")
print("=" * 60)

try:
    model = YOLO('yolov8n.pt')
    model.overrides['verbose'] = False  # Desabilita prints do YOLO
    print("✓ YOLO v8 Nano carregado com sucesso!")
except Exception as e:
    print(f"✗ Erro ao carregar YOLO: {e}")
    sys.exit(1)

# Classes COCO que o YOLO detecta
COCO_CLASSES = model.names

# ==================== CLASSE PRINCIPAL ====================
class YOLOCamera:
    def __init__(self, camera_index=0):
        self.video = cv2.VideoCapture(camera_index)
        
        if not self.video.isOpened():
            raise Exception("Erro ao abrir a câmera!")
        
        # Configurações da câmera
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        # Obtem propriedades reais
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Câmera: {self.width}x{self.height} @ {self.fps}fps")
        
        # Controle de FPS
        self.fps_start_time = datetime.now()
        self.fps_frame_count = 0
        self.current_fps = 0

    def __del__(self):
        self.video.release()

    def process_frame(self, frame):
        """Processa frame com YOLO"""
        
        # Detecção YOLO (desabilita visualização automática)
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, show=False, stream=False)
        
        # Contadores
        person_count = 0
        object_count = 0
        
        # Processa detecções YOLO
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = COCO_CLASSES[cls]
                
                # Conta pessoas
                if class_name == 'person':
                    person_count += 1
                    color = (0, 255, 0)  # Verde para pessoas
                else:
                    object_count += 1
                    color = (255, 165, 0)  # Laranja para objetos
                
                # Desenha box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame, person_count, object_count

    def calculate_fps(self):
        """Calcula FPS atual"""
        self.fps_frame_count += 1
        elapsed = (datetime.now() - self.fps_start_time).total_seconds()
        
        if elapsed > 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = datetime.now()
        
        return self.current_fps

    def draw_info(self, frame, person_count, object_count):
        """Desenha informações na tela"""
        info_y = 30
        line_height = 30
        
        # Background semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Informações
        infos = [
            f"Pessoas: {person_count}",
            f"Objetos: {object_count}",
        ]
        
        if SHOW_FPS:
            infos.append(f"FPS: {self.current_fps:.1f}")
        
        for i, info in enumerate(infos):
            cv2.putText(frame, info, (10, info_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Controle
        cv2.putText(frame, "Pressione 'Q' para sair", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def get_frame(self):
        """Captura e processa frame"""
        ret, frame = self.video.read()
        
        if not ret:
            return None
        
        # Processa detecções
        frame, person_count, object_count = self.process_frame(frame)
        
        # Calcula FPS
        if SHOW_FPS:
            self.calculate_fps()
        
        # Desenha informações
        frame = self.draw_info(frame, person_count, object_count)
        
        return frame


# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    print("\n" + "=" * 60)
    print("SISTEMA DE DETECÇÃO COM YOLO v8")
    print("=" * 60)
    print(f"\nConectando à câmera (índice {CAMERA_INDEX})...")
    
    try:
        camera = YOLOCamera(CAMERA_INDEX)
    except Exception as e:
        print(f"\n✗ ERRO: {e}")
        print("\nDicas:")
        print("- Verifique se o Iriun Webcam está rodando")
        print("- Tente outros índices: 1, 2, 3...")
        print("- Execute: v4l2-ctl --list-devices")
        return
    
    print("\n" + "=" * 60)
    print("SISTEMA PRONTO!")
    print("=" * 60)
    print("\nPressione 'Q' para sair")
    print("=" * 60 + "\n")
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("✗ Falha ao capturar frame")
                break
            
            # Mostra apenas UMA janela
            cv2.imshow('YOLO v8 - Deteccao', frame)
            
            # Pressione 'q' ou ESC para sair
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    
    except KeyboardInterrupt:
        print("\n\n✓ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n✗ Erro: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print("\n✓ Sistema encerrado\n")


if __name__ == "__main__":
    main()