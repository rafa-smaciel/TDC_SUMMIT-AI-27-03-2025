import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import queue
import math

# IMPORT para data/hora com timezone
from datetime import datetime
import pytz

# IMPORT para o MongoDB (PyMongo)
from pymongo import MongoClient

class VideoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análise de Vídeo - TDC Summit")
        self.configure(bg="#2e2e2e")
        self.state("zoomed")  # Abre maximizado

        try:
            self.window_icon = ImageTk.PhotoImage(file="logo.png")
            self.iconphoto(False, self.window_icon)
        except Exception as e:
            print("Erro ao carregar o ícone:", e)

        # Filas para logs e frames de vídeo
        self.log_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.piece_count = 0

        self.video_path = None
        self.model_path = None

        # Variável para controle do loop do vídeo
        self.loop_video = tk.BooleanVar(value=False)

        # Conexão com o MongoDB
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017/")
            self.db = self.mongo_client["tdc_workshop"]
            self.logs_collection = self.db["event_logs"]
            self.log("Conectado ao MongoDB com sucesso.")
        except Exception as e:
            self.log(f"Erro ao conectar ao MongoDB: {e}")

        # Interface principal
        self.main_frame = tk.Frame(self, bg="#2e2e2e")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Menu lateral
        self.menu_frame = tk.Frame(self.main_frame, width=300, bg="#3e3e3e")
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y)

        try:
            self.logo_img = Image.open("tdc.png")
            self.logo_img = self.logo_img.resize((200, 60), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
            self.logo_label = tk.Label(self.menu_frame, image=self.logo_photo, bg="#3e3e3e")
            self.logo_label.pack(pady=(20, 5))
        except Exception as e:
            self.logo_label = tk.Label(self.menu_frame, text="Vega Robotics", font=("Arial", 20, "bold"),
                                        bg="#3e3e3e", fg="white")
            self.logo_label.pack(pady=(20, 5))

        # Botões de seleção com "✔" à esquerda
        self.frame_model = tk.Frame(self.menu_frame, bg="#3e3e3e")
        self.frame_model.pack(pady=10, fill=tk.X, padx=20)
        self.model_ok_label = tk.Label(self.frame_model, text="", bg="#3e3e3e", fg="lime", font=("Arial", 14))
        self.model_ok_label.pack(side=tk.LEFT)
        self.btn_select_model = tk.Button(self.frame_model, text="Selecionar Modelo", command=self.select_model,
                                          bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_select_model.pack(side=tk.LEFT, padx=5)

        self.frame_video = tk.Frame(self.menu_frame, bg="#3e3e3e")
        self.frame_video.pack(pady=10, fill=tk.X, padx=20)
        self.video_ok_label = tk.Label(self.frame_video, text="", bg="#3e3e3e", fg="lime", font=("Arial", 14))
        self.video_ok_label.pack(side=tk.LEFT)
        self.btn_select_video = tk.Button(self.frame_video, text="Selecionar Vídeo", command=self.select_video,
                                          bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_select_video.pack(side=tk.LEFT, padx=5)

        # Checkbutton para ativar o loop do vídeo com cor de seleção definida
        self.chk_loop = tk.Checkbutton(
            self.menu_frame,
            text="Loop do Vídeo",
            variable=self.loop_video,
            command=self.print_loop_state,
            bg="#3e3e3e",
            fg="white",
            activebackground="#5a5a5a",
            selectcolor="lightblue"  # Torna a marcação visível
        )
        self.chk_loop.pack(pady=10, fill=tk.X, padx=20)

        # Botão Executar
        self.btn_run = tk.Button(self.menu_frame, text="Executar", command=self.run_detection,
                                 bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_run.pack(pady=10, fill=tk.X, padx=20)

        # Botão Ocultar Aplicativo
        self.btn_hide = tk.Button(self.menu_frame, text="Ocultar Aplicativo", command=self.hide_app,
                                  bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_hide.pack(pady=10, fill=tk.X, padx=20)

        # Área de conteúdo: vídeo e terminal
        self.content_frame = tk.Frame(self.main_frame, bg="#2e2e2e")
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_frame = tk.Frame(self.content_frame, bg='black')
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.terminal_frame = tk.Frame(self.content_frame, bg='black', width=300)
        self.terminal_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.terminal_text = tk.Text(self.terminal_frame, bg='black', fg='lime', font=('Courier', 10))
        self.terminal_text.pack(fill=tk.BOTH, expand=True)
        self.terminal_text.insert(tk.END, "[TERMINAL] Pronto para iniciar...\n")
        self.terminal_text.config(state=tk.DISABLED)

        # Inicia os loops de atualização para logs e vídeo
        self.after(100, self.process_log_queue)
        self.after(30, self.update_video)

    def print_loop_state(self):
        # Função de debug para verificar o estado do loop
        print("Loop habilitado:", self.loop_video.get())

    def log(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.terminal_text.config(state=tk.NORMAL)
                self.terminal_text.insert(tk.END, msg + "\n")
                self.terminal_text.see(tk.END)
                self.terminal_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)

    def update_video(self):
        try:
            while True:
                imgtk = self.video_queue.get_nowait()
                self.video_label.configure(image=imgtk)
                self.video_label.imgtk = imgtk
        except queue.Empty:
            pass
        self.after(30, self.update_video)

    def select_model(self):
        path = filedialog.askopenfilename(title="Selecione o Modelo YOLO",
                                          filetypes=[("Arquivos .pt", "*.pt")])
        if path:
            self.model_path = path
            self.model_ok_label.config(text="✔")
            self.log(f"Modelo selecionado: {os.path.basename(path)}")

    def select_video(self):
        path = filedialog.askopenfilename(title="Selecione o Vídeo",
                                          filetypes=[("Vídeos", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.video_ok_label.config(text="✔")
            self.log(f"Vídeo selecionado: {os.path.basename(path)}")

    def run_detection(self):
        if not self.model_path or not self.video_path:
            messagebox.showerror("Erro", "Selecione o modelo e o vídeo primeiro.")
            return
        self.log("Iniciando detecção de objetos...")
        threading.Thread(target=self.detect_objects, daemon=True).start()

    def detect_objects(self):
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_path)

        # Rastreamento simples: guarda objetos detectados com posição atual e anterior
        objetos = {}  # key: id, value: {"pos": (x, y), "prev_pos": (x, y), "contado": bool, "classe": str, "lost": int}
        next_id = 0

        # Parâmetros de tracking
        tolerancia = 60    # Tolerância para associação (em pixels)
        max_lost = 5       # Frames sem detecção para remover objeto

        while True:
            ret, frame = cap.read()
            if not ret:
                # Se o loop estiver ativado, reinicia a captura
                if self.loop_video.get():
                    cap.release()
                    cap = cv2.VideoCapture(self.video_path)
                    continue
                else:
                    break

            h_frame, w_frame = frame.shape[:2]
            linha_meio = h_frame // 2  # Linha central do frame original

            # Inferência com YOLO
            results = model.predict(frame, verbose=False)
            deteccoes = []  # Lista de ((cx, cy), classe)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    classe = "Aprovado" if cls == 0 else "Reprovado"

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    deteccoes.append(((cx, cy), classe))

                    # Desenho no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, classe, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                    # Exibe a porcentagem como inteiro
                    percent = ((cy - linha_meio) / h_frame) * 100
                    percent_int = int(round(percent))
                    cv2.putText(frame, f"{percent_int}%", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Atualiza a posição anterior de cada objeto rastreado
            for obj in objetos.values():
                obj["prev_pos"] = obj["pos"]

            # Associa detecções a objetos existentes
            ids_atualizados = set()
            for (cx, cy), classe in deteccoes:
                melhor_id, melhor_dist = None, tolerancia + 1
                for obj_id, obj in objetos.items():
                    dist = math.hypot(cx - obj["pos"][0], cy - obj["pos"][1])
                    if dist < tolerancia and dist < melhor_dist:
                        melhor_dist = dist
                        melhor_id = obj_id

                if melhor_id is not None:
                    objetos[melhor_id]["pos"] = (cx, cy)
                    objetos[melhor_id]["classe"] = classe
                    objetos[melhor_id]["lost"] = 0
                    ids_atualizados.add(melhor_id)
                else:
                    objetos[next_id] = {
                        "pos": (cx, cy),
                        "prev_pos": (cx, cy),
                        "contado": False,
                        "classe": classe,
                        "lost": 0
                    }
                    ids_atualizados.add(next_id)
                    next_id += 1

            # Incrementa "lost" para objetos não atualizados
            for obj_id in list(objetos.keys()):
                if obj_id not in ids_atualizados:
                    objetos[obj_id]["lost"] += 1
                    if objetos[obj_id]["lost"] > max_lost:
                        del objetos[obj_id]

            # Verifica se o objeto atingiu 0% (apenas uma vez por objeto)
            for obj in objetos.values():
                y_prev = obj["prev_pos"][1]
                y_atual = obj["pos"][1]
                percent_prev = ((y_prev - linha_meio) / h_frame) * 100
                percent_atual = ((y_atual - linha_meio) / h_frame) * 100
                percent_atual_int = int(round(percent_atual))

                # Debug no console
                print(f"Objeto {obj['classe']} - Prev: {percent_prev:.2f}%, Atual: {percent_atual:.2f}%")

                if not obj["contado"] and percent_atual_int == 0:
                    obj["contado"] = True
                    self.piece_count += 1

                    # Obtém data/hora atual no fuso de Brasília
                    fuso_br = pytz.timezone("America/Sao_Paulo")
                    agora_br = datetime.now(fuso_br)
                    data_hora_str = agora_br.strftime("%d/%m/%Y %H:%M:%S")

                    # Registra o evento no terminal
                    self.log(f"{obj['classe']}, Placa Sextavada, 1318, {data_hora_str} | Total: {self.piece_count}")

                    # Persiste o evento no MongoDB
                    self.logs_collection.insert_one({
                        "classe": obj["classe"],
                        "nome_item": "Placa Sextavada",
                        "codigo": "1318",
                        "data_hora": agora_br,
                        "total": self.piece_count
                    })

            # Desenha a linha de detecção (vermelha)
            cv2.line(frame, (0, linha_meio), (w_frame, linha_meio), (0, 0, 255), 2)

            # Redimensiona o frame para exibição mantendo a razão de aspecto
            self.video_frame.update_idletasks()
            w_cont = self.video_frame.winfo_width()
            h_cont = self.video_frame.winfo_height()
            escala = min(w_cont / w_frame, h_cont / h_frame)
            novo_w = int(w_frame * escala)
            novo_h = int(h_frame * escala)
            frame_resized = cv2.resize(frame, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

            # Converte para exibição no Tkinter
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_queue.put(imgtk)

        cap.release()
        self.log("Processamento finalizado.")

    def hide_app(self):
        self.withdraw()
        self.esc_listener = tk.Toplevel()
        self.esc_listener.overrideredirect(True)
        self.esc_listener.geometry("300x50+100+100")
        self.esc_listener.config(bg="#2e2e2e")
        info_label = tk.Label(self.esc_listener, text="Pressione ESC para restaurar o aplicativo",
                              bg="#2e2e2e", fg="white", font=("Arial", 10))
        info_label.pack(expand=True, fill="both")
        self.esc_listener.bind("<Escape>", self.show_app)
        self.esc_listener.focus_force()

    def show_app(self, event=None):
        self.deiconify()
        if hasattr(self, "esc_listener"):
            self.esc_listener.destroy()

if __name__ == "__main__":
    app = VideoApp()
    app.mainloop()
