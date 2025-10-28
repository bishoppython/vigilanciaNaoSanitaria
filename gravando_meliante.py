#!/usr/bin/env python3

"""
Sistema de Detecção com YOLO v8 + Detecção de Movimento
- Detecção de pessoas e objetos (YOLO v8)
- Gravação apenas para movimentos significativos de pessoas
- Ignora movimentos sutis (vento, pequenas oscilações)
"""

import cv2
import sys
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ==================== CONFIGURAÇÕES ====================
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5  # Confiança mínima para detecção YOLO
SHOW_FPS = True             # Mostrar FPS na tela
RECORD_ON_MOVEMENT = True   # Gravar quando detectar movimento de pessoas
MIN_PERSON_COUNT = 1        # Número mínimo de pessoas para considerar
RECORD_DURATION_AFTER_LAST_MOVEMENT = 5  # Segundos extras de gravação
OUTPUT_DIR = "recordings"   # Pasta para salvar os vídeos
RECORD_FPS = 10             # FPS FIXO para gravação

# ==================== CONFIGURAÇÕES DE MOVIMENTO OTIMIZADAS ====================
# Configurações para detectar apenas movimentos significativos
MIN_MOTION_AREA = 2500      # AUMENTADO: Área mínima de pixels em movimento (movimentos maiores)
MOTION_DURATION_THRESHOLD = 8  # Número de frames consecutivos com movimento necessário
MOTION_CONFIDENCE_THRESHOLD = 60  # AUMENTADO: % mínima de confiança para considerar movimento
AREA_COVERAGE_THRESHOLD = 0.15  # Mínimo 15% da área da pessoa deve ter movimento

# Configurações do background subtractor para menos sensibilidade
BG_HISTORY = 300           # Histórico menor para se adaptar mais rápido
BG_THRESHOLD = 25          # AUMENTADO: Menos sensível a mudanças sutis
LEARNING_RATE = 0.005      # AUMENTADO: Aprende mais rápido o background

# ==================== CLASSE PRINCIPAL ====================
class YOLOCamera:
    def __init__(self, camera_index=0):
        self.video = cv2.VideoCapture(camera_index)
        
        if not self.video.isOpened():
            raise Exception("Erro ao abrir a câmera!")
        
        # Configurações da câmera
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        # Obtem propriedades reais
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Câmera: {self.width}x{self.height} @ {self.fps}fps")
        print(f"✓ FPS de gravação: {RECORD_FPS} (FIXO)")
        print(f"✓ Configurações de movimento:")
        print(f"  - Área mínima: {MIN_MOTION_AREA} pixels")
        print(f"  - Frames consecutivos: {MOTION_DURATION_THRESHOLD}")
        print(f"  - Confiança mínima: {MOTION_CONFIDENCE_THRESHOLD}%")
        
        # Sistema de detecção de movimento OTIMIZADO
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY, 
            varThreshold=BG_THRESHOLD, 
            detectShadows=False  # DESABILITADO: ignora sombras
        )
        self.motion_detected = False
        self.motion_confidence = 0
        self.consecutive_motion_frames = 0
        self.consecutive_no_motion_frames = 0
        self.motion_regions = []
        self.significant_motion_detected = False
        
        # Controle de FPS para display
        self.fps_start_time = datetime.now()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Sistema de gravação
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.last_movement_time = None
        self.current_video_filename = None
        self.last_record_time = datetime.now()
        self.record_interval = 1.0 / RECORD_FPS
        self.frame_count = 0
        
        # Cria diretório de gravações
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"✓ Diretório '{OUTPUT_DIR}' criado")

    def detect_significant_movement(self, frame, person_boxes):
        """Detecta apenas movimentos significativos de pessoas"""
        if len(person_boxes) == 0:
            self.significant_motion_detected = False
            self.motion_confidence = 0
            self.consecutive_motion_frames = 0
            return False, 0, []

        # Converte para escala de cinza com menos blur (para manter detalhes de movimento)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)  # Blur reduzido
        
        # Aplica subtração de background menos sensível
        fg_mask = self.background_subtractor.apply(gray, learningRate=LEARNING_RATE)
        
        # Operações morfológicas mais agressivas para remover ruídos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)  # Dilatação reduzida
        
        # Encontra contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_motion_detected = False
        motion_regions = []
        total_significant_area = 0
        
        # Analisa cada contorno de movimento
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            
            # FILTRO 1: Área mínima aumentada para movimentos significativos
            if contour_area < MIN_MOTION_AREA:
                continue
                
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_region = (x, y, x + w, y + h)
            
            # Verifica se o movimento está dentro de alguma bounding box de pessoa
            for person_box in person_boxes:
                px1, py1, px2, py2 = person_box
                person_area = (px2 - px1) * (py2 - py1)
                
                # Calcula interseção entre região de movimento e bounding box da pessoa
                intersection_x1 = max(x, px1)
                intersection_y1 = max(y, py1)
                intersection_x2 = min(x + w, px2)
                intersection_y2 = min(y + h, py2)
                
                if (intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1):
                    # Calcula área de interseção
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    
                    # FILTRO 2: Pelo menos X% da área da pessoa deve estar em movimento
                    area_coverage = intersection_area / person_area
                    if area_coverage >= AREA_COVERAGE_THRESHOLD:
                        significant_motion_detected = True
                        motion_regions.append(motion_region)
                        total_significant_area += contour_area
                        break
        
        # FILTRO 3: Requer múltiplos frames consecutivos com movimento
        if significant_motion_detected:
            self.consecutive_motion_frames = min(self.consecutive_motion_frames + 1, 30)
            self.consecutive_no_motion_frames = 0
        else:
            self.consecutive_motion_frames = max(self.consecutive_motion_frames - 2, 0)
            self.consecutive_no_motion_frames += 1
        
        # Calcula confiança baseada em área e duração
        if self.consecutive_motion_frames > 0:
            area_confidence = min(total_significant_area / 10000, 1.0)  # 10000 pixels = 100%
            duration_confidence = min(self.consecutive_motion_frames / MOTION_DURATION_THRESHOLD, 1.0)
            self.motion_confidence = int((area_confidence * 0.6 + duration_confidence * 0.4) * 100)
        else:
            self.motion_confidence = max(self.motion_confidence - 15, 0)
        
        # FILTRO 4: Só considera movimento significativo se passar todos os thresholds
        self.significant_motion_detected = (
            self.consecutive_motion_frames >= MOTION_DURATION_THRESHOLD and 
            self.motion_confidence >= MOTION_CONFIDENCE_THRESHOLD
        )
        
        self.motion_regions = motion_regions
        
        return self.significant_motion_detected, self.motion_confidence, motion_regions

    def start_recording(self):
        """Inicia a gravação do vídeo"""
        if self.is_recording:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_filename = f"{OUTPUT_DIR}/movement_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_video_filename,
            fourcc,
            RECORD_FPS,
            (self.width, self.height)
        )
        
        if not self.video_writer.isOpened():
            print("✗ Erro ao inicializar VideoWriter")
            self.video_writer = None
            return
        
        self.is_recording = True
        self.recording_start_time = datetime.now()
        self.last_movement_time = datetime.now()
        self.last_record_time = datetime.now()
        self.frame_count = 0
        
        print(f"🎥 Gravação iniciada por MOVIMENTO SIGNIFICATIVO")
        print(f"   → Arquivo: {self.current_video_filename}")
        print(f"   → Confiança: {self.motion_confidence}%")
        print(f"   → Frames consecutivos: {self.consecutive_motion_frames}")

    def should_record_frame(self):
        """Verifica se deve gravar o frame atual"""
        if not self.is_recording:
            return False
            
        current_time = datetime.now()
        time_since_last_record = (current_time - self.last_record_time).total_seconds()
        
        if time_since_last_record >= self.record_interval:
            self.last_record_time = current_time
            self.frame_count += 1
            return True
        return False

    def stop_recording(self):
        """Para a gravação do vídeo"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            duration = (datetime.now() - self.recording_start_time).total_seconds()
            actual_fps = self.frame_count / duration if duration > 0 else 0
            print(f"⏹️  Gravação finalizada: {duration:.1f}s, {actual_fps:.1f} FPS real")
            self.is_recording = False
            self.current_video_filename = None

    def should_stop_recording(self):
        """Verifica se deve parar a gravação"""
        if not self.is_recording or self.last_movement_time is None:
            return False
            
        time_since_last_movement = (datetime.now() - self.last_movement_time).total_seconds()
        return time_since_last_movement > RECORD_DURATION_AFTER_LAST_MOVEMENT

    def process_frame(self, frame):
        """Processa frame com YOLO e detecção de movimento significativo"""
        # Detecção YOLO
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, show=False, stream=False)
        
        # Contadores e bounding boxes de pessoas
        person_count = 0
        object_count = 0
        person_boxes = []
        
        # Processa detecções YOLO
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = COCO_CLASSES[cls]
                
                if class_name == 'person':
                    person_count += 1
                    color = (0, 255, 0)
                    person_boxes.append((x1, y1, x2, y2))
                else:
                    object_count += 1
                    color = (255, 165, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Detecção de movimento significativo
        movement_detected = False
        movement_confidence = 0
        movement_regions = []
        
        if person_count >= MIN_PERSON_COUNT:
            movement_detected, movement_confidence, movement_regions = self.detect_significant_movement(frame, person_boxes)
            
            # Desenha regiões de movimento significativo
            for (mx1, my1, mx2, my2) in movement_regions:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 3)
                cv2.putText(frame, "MOVIMENTO!", (mx1, my1 - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Controle de gravação baseado em movimento significativo
        if RECORD_ON_MOVEMENT:
            if movement_detected and person_count >= MIN_PERSON_COUNT:
                self.last_movement_time = datetime.now()
                if not self.is_recording:
                    self.start_recording()
            else:
                if self.is_recording and self.should_stop_recording():
                    self.stop_recording()
        
        return frame, person_count, object_count, movement_detected, movement_confidence

    def calculate_fps(self):
        """Calcula FPS atual para display"""
        self.fps_frame_count += 1
        elapsed = (datetime.now() - self.fps_start_time).total_seconds()
        
        if elapsed > 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = datetime.now()
        
        return self.current_fps

    def draw_info(self, frame, person_count, object_count, movement_detected, movement_confidence):
        """Desenha informações na tela"""
        info_y = 30
        line_height = 25
        
        # Background semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (500, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Informações
        infos = [
            f"Pessoas: {person_count}",
            f"Movimento: {'SIGNIFICATIVO!' if movement_detected else 'insignificante'}",
            f"Confianca: {movement_confidence}%",
            f"Frames Consecutivos: {self.consecutive_motion_frames}/{MOTION_DURATION_THRESHOLD}",
            f"FPS Display: {self.current_fps:.1f}",
            f"FPS Gravacao: {RECORD_FPS}",
        ]
        
        # Status de gravação
        if self.is_recording:
            recording_time = (datetime.now() - self.recording_start_time).total_seconds()
            infos.append(f"🎥 GRAVANDO: {recording_time:.1f}s")
            infos.append(f"📁 {os.path.basename(self.current_video_filename)}")
        
        for i, info in enumerate(infos):
            color = (0, 255, 0)  # Verde padrão
            if "GRAVANDO" in info:
                color = (0, 0, 255)  # Vermelho
            elif "SIGNIFICATIVO" in info:
                color = (0, 255, 255)  # Amarelo forte
            elif "insignificante" in info:
                color = (150, 150, 150)  # Cinza para movimento insignificante
            elif "Confianca" in info:
                if movement_confidence >= 80:
                    color = (0, 255, 0)  # Verde alto
                elif movement_confidence >= 60:
                    color = (0, 255, 255)  # Amarelo
                else:
                    color = (0, 165, 255)  # Laranja
            cv2.putText(frame, info, (10, info_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Legenda
        cv2.putText(frame, "Sistema: Ignora vento/roupas balançando, detecta pessoas se movendo", 
                   (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Indicador visual
        if self.is_recording:
            cv2.circle(frame, (self.width - 20, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 50, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

    def get_frame(self):
        """Captura e processa frame"""
        ret, frame = self.video.read()
        
        if not ret:
            return None
        
        # Processa detecções
        frame, person_count, object_count, movement_detected, movement_confidence = self.process_frame(frame)
        
        # Escreve frame no vídeo
        if self.is_recording and self.video_writer is not None and self.should_record_frame():
            self.video_writer.write(frame)
        
        # Calcula FPS para display
        if SHOW_FPS:
            self.calculate_fps()
        
        # Desenha informações
        frame = self.draw_info(frame, person_count, object_count, movement_detected, movement_confidence)
        
        return frame


# ==================== CARREGAMENTO DO YOLO ====================
print("=" * 60)
print("Carregando modelo YOLO v8...")
print("=" * 60)

try:
    model = YOLO('yolov8n.pt')
    model.overrides['verbose'] = False
    print("✓ YOLO v8 Nano carregado com sucesso!")
except Exception as e:
    print(f"✗ Erro ao carregar YOLO: {e}")
    sys.exit(1)

COCO_CLASSES = model.names

# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    print("\n" + "=" * 60)
    print("SISTEMA DE DETECÇÃO - MOVIMENTOS SIGNIFICATIVOS")
    print("=" * 60)
    print(f"\n⚡ CONFIGURAÇÕES OTIMIZADAS:")
    print(f"  - Área mínima movimento: {MIN_MOTION_AREA} pixels")
    print(f"  - Frames consecutivos necessários: {MOTION_DURATION_THRESHOLD}")
    print(f"  - Confiança mínima: {MOTION_CONFIDENCE_THRESHOLD}%")
    print(f"  - Cobertura área pessoa: {AREA_COVERAGE_THRESHOLD*100}%")
    print(f"\n🎯 COMPORTAMENTO ESPERADO:")
    print(f"  ✓ DETECTA: Pessoas andando, se movendo, gesticulando")
    print(f"  ✗ IGNORA: Vento balançando roupas, pequenas oscilações")
    print(f"  ✗ IGNORA: Movimentos sutis, sombras, ruídos")
    
    try:
        camera = YOLOCamera(CAMERA_INDEX)
    except Exception as e:
        print(f"\n✗ ERRO: {e}")
        return
    
    print("\n" + "=" * 60)
    print("SISTEMA PRONTO PARA DETECÇÃO DE MOVIMENTOS REAIS!")
    print("=" * 60)
    print("\nPressione 'Q' para sair")
    print("=" * 60 + "\n")
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("✗ Falha ao capturar frame")
                break
            
            cv2.imshow('Detector - Movimentos Significativos', frame)
            
            # key = cv2.waitKey(1) & 0xFF
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
    
    except KeyboardInterrupt:
        print("\n\n✓ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n✗ Erro: {e}")
    
    finally:
        camera.stop_recording()
        cv2.destroyAllWindows()
        print("\n✓ Sistema encerrado\n")


if __name__ == "__main__":
    main()