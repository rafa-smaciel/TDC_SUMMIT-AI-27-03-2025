import os
import threading
import time
import queue
import math
import json
import csv
from datetime import datetime
import pytz
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # Para manipulação de imagens
import cv2
from pymongo import MongoClient
from ultralytics import YOLO

# ========================================================
# ## 1. AGENTES - Integração com a interface via callback de log
# ========================================================

class BaseAgent(threading.Thread):
    """
    Classe base para os agentes. Herda de threading.Thread e implementa
    um ciclo de vida padrão (iniciar, escutar e parar).
    """
    def __init__(self, name, log_callback=None):
        super().__init__(daemon=True)
        self.name = name
        self.running = False
        self.log_callback = log_callback

    def run(self):
        self.running = True
        if self.log_callback:
            self.log_callback(f"[{self.name}] Iniciado.")
        while self.running:
            self.listen()
            time.sleep(1)

    def listen(self):
        raise NotImplementedError("Subclasse deve implementar listen().")

    def stop(self):
        self.running = False


class SupervisorAgent(BaseAgent):
    """
    Agente Supervisor: Processa os eventos e, se o evento for 'Reprovado',
    dispara um alerta crítico (através do alert_callback).
    """
    def __init__(self, log_callback=None, alert_callback=None):
        super().__init__("Supervisor", log_callback)
        self.alert_callback = alert_callback

    def process_event(self, event):
        # Se o evento for crítico (classe 'Reprovado'), dispara o alerta
        if event.get("classe") == "Reprovado":
            msg = f"[Supervisor] ⚠️ Evento CRÍTICO detectado: {event}"
            if self.alert_callback:
                self.alert_callback(msg)
        else:
            msg = f"[Supervisor] ✅ Evento aprovado ignorado."
        if self.log_callback:
            self.log_callback(msg)

    def listen(self):
        pass  # Atua apenas via callback (não há escuta contínua)


class EventMonitorAgent(BaseAgent):
    """
    Agente Monitor: Conecta ao MongoDB e verifica a coleção 'event_logs'
    para detectar novos eventos. Quando detecta, chama o callback configurado.
    """
    def __init__(self, callback, log_callback=None):
        super().__init__("Monitor", log_callback)
        self.mongo = MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo["tdc_workshop"]
        self.collection = self.db["event_logs"]
        self.last_check = datetime.utcnow()
        self.callback = callback

    def listen(self):
        novos = list(self.collection.find({"data_hora": {"$gt": self.last_check}}))
        if novos:
            self.last_check = datetime.utcnow()
            for e in novos:
                msg = f"[Monitor] Novo evento detectado: {e}"
                if self.log_callback:
                    self.log_callback(msg)
                else:
                    print(msg)
                if self.callback:
                    self.callback(e)

# ========================================================
# ## 2. APLICATIVO DE VÍDEO COM INTEGRAÇÃO DOS AGENTES
# ========================================================

class VideoApp(tk.Tk):
    """
    Aplicativo principal que integra a interface gráfica com os agentes.
    Possui:
      - Painel de status para os agentes
      - Área para exibição do vídeo e logs
      - Botões para selecionar modelo, vídeo e executar a detecção
      - Pop-up de alerta crítico com ícone PNG
    """
    def __init__(self):
        super().__init__()
        self.title("Copilot 27/03/2025 - TDC Summit AI")
        self.configure(bg="#2e2e2e")
        self.state("zoomed")  # Abre a janela maximizada

        # ============================
        # ### Configuração do Ícone da Janela
        # ============================
        try:
            self.window_icon = ImageTk.PhotoImage(file="logo.png")
            self.iconphoto(False, self.window_icon)
        except Exception as e:
            print("Erro ao carregar o ícone:", e)

        # ============================
        # ### Inicialização de Filas e Variáveis
        # ============================
        self.log_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.piece_count = 0
        self.video_path = None
        self.model_path = None
        self.loop_video = tk.BooleanVar(value=False)

        # ============================
        # ### Conexão com o MongoDB
        # ============================
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017/")
            self.db = self.mongo_client["tdc_workshop"]
            self.logs_collection = self.db["event_logs"]
            self.log("Conectado ao MongoDB com sucesso.")
        except Exception as e:
            self.log(f"Erro ao conectar ao MongoDB: {e}")

        # ============================
        # ### Configuração da Interface Principal
        # ============================
        self.main_frame = tk.Frame(self, bg="#2e2e2e")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # ----- Menu Lateral -----
        self.menu_frame = tk.Frame(self.main_frame, width=300, bg="#3e3e3e")
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y)

        # >>> Painel de Status dos Agentes <<<
        self.status_frame = tk.Frame(self.menu_frame, bg="#3e3e3e")
        self.status_frame.pack(pady=10, fill=tk.X, padx=20)
        status_title = tk.Label(self.status_frame, text="Status dos Agentes", bg="#3e3e3e",
                                fg="white", font=("Arial", 12, "bold"))
        status_title.pack(anchor="w")

        # --- Status do Supervisor ---
        self.supervisor_status_frame = tk.Frame(self.status_frame, bg="#3e3e3e")
        self.supervisor_status_frame.pack(anchor="w", pady=2)
        self.supervisor_canvas = tk.Canvas(self.supervisor_status_frame, width=12, height=12,
                                            bg="#3e3e3e", highlightthickness=0)
        self.supervisor_canvas.pack(side=tk.LEFT, padx=(0, 5))
        self.supervisor_circle = self.supervisor_canvas.create_oval(2, 2, 10, 10, fill="red", outline="red")
        self.supervisor_label = tk.Label(self.supervisor_status_frame, text="Supervisor: Inativo",
                                         bg="#3e3e3e", fg="red", font=("Arial", 10))
        self.supervisor_label.pack(side=tk.LEFT)

        # --- Status do Monitor ---
        self.monitor_status_frame = tk.Frame(self.status_frame, bg="#3e3e3e")
        self.monitor_status_frame.pack(anchor="w", pady=2)
        self.monitor_canvas = tk.Canvas(self.monitor_status_frame, width=12, height=12,
                                         bg="#3e3e3e", highlightthickness=0)
        self.monitor_canvas.pack(side=tk.LEFT, padx=(0, 5))
        self.monitor_circle = self.monitor_canvas.create_oval(2, 2, 10, 10, fill="red", outline="red")
        self.monitor_label = tk.Label(self.monitor_status_frame, text="Monitor: Inativo",
                                      bg="#3e3e3e", fg="red", font=("Arial", 10))
        self.monitor_label.pack(side=tk.LEFT)

        # Inicia a atualização dos status dos agentes
        self.update_agent_status()

        # ----- Elementos do Menu Lateral -----
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

        self.frame_model = tk.Frame(self.menu_frame, bg="#3e3e3e")
        self.frame_model.pack(pady=10, fill=tk.X, padx=20)
        self.model_ok_label = tk.Label(self.frame_model, text="", bg="#3e3e3e",
                                       fg="lime", font=("Arial", 14))
        self.model_ok_label.pack(side=tk.LEFT)
        self.btn_select_model = tk.Button(self.frame_model, text="Selecionar Modelo",
                                          command=self.select_model,
                                          bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_select_model.pack(side=tk.LEFT, padx=5)

        self.frame_video = tk.Frame(self.menu_frame, bg="#3e3e3e")
        self.frame_video.pack(pady=10, fill=tk.X, padx=20)
        self.video_ok_label = tk.Label(self.frame_video, text="", bg="#3e3e3e",
                                       fg="lime", font=("Arial", 14))
        self.video_ok_label.pack(side=tk.LEFT)
        self.btn_select_video = tk.Button(self.frame_video, text="Selecionar Vídeo",
                                          command=self.select_video,
                                          bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_select_video.pack(side=tk.LEFT, padx=5)

        self.chk_loop = tk.Checkbutton(
            self.menu_frame,
            text="Loop do Vídeo",
            variable=self.loop_video,
            command=self.print_loop_state,
            bg="#3e3e3e", fg="white", activebackground="#5a5a5a",
            selectcolor="lightblue"
        )
        self.chk_loop.pack(pady=10, fill=tk.X, padx=20)

        self.btn_run = tk.Button(self.menu_frame, text="Executar", command=self.run_detection,
                                 bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_run.pack(pady=10, fill=tk.X, padx=20)

        self.btn_hide = tk.Button(self.menu_frame, text="Ocultar Aplicativo", command=self.hide_app,
                                  bg="#3e3e3e", fg="white", activebackground="#5a5a5a")
        self.btn_hide.pack(pady=10, fill=tk.X, padx=20)

        # ----- Nota: Botão de exportação removido conforme solicitado -----

        # ----- Área de Conteúdo: Vídeo e Terminal -----
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

        # ----- Inicia os ciclos de atualização -----
        self.after(100, self.process_log_queue)
        self.after(30, self.update_video)

        # ============================
        # ### Instancia e inicia os Agentes
        # ============================
        self.supervisor_agent = SupervisorAgent(log_callback=self.log,
                                                alert_callback=self.show_critical_alert)
        self.monitor_agent = EventMonitorAgent(callback=self.supervisor_agent.process_event,
                                                log_callback=self.log)
        self.supervisor_agent.start()
        self.monitor_agent.start()

    # ========================================================
    # ## 3. Métodos de Atualização e Logs
    # ========================================================

    def update_agent_status(self):
        """Atualiza o status visual dos agentes (Supervisor e Monitor) no painel."""
        if hasattr(self, 'supervisor_agent'):
            if self.supervisor_agent.running:
                self.supervisor_canvas.itemconfig(self.supervisor_circle, fill="lime", outline="lime")
                self.supervisor_label.config(text="Supervisor: Ativo", fg="lime")
            else:
                self.supervisor_canvas.itemconfig(self.supervisor_circle, fill="red", outline="red")
                self.supervisor_label.config(text="Supervisor: Inativo", fg="red")
        if hasattr(self, 'monitor_agent'):
            if self.monitor_agent.running:
                self.monitor_canvas.itemconfig(self.monitor_circle, fill="lime", outline="lime")
                self.monitor_label.config(text="Monitor: Ativo", fg="lime")
            else:
                self.monitor_canvas.itemconfig(self.monitor_circle, fill="red", outline="red")
                self.monitor_label.config(text="Monitor: Inativo", fg="red")
        self.after(1000, self.update_agent_status)

    def print_loop_state(self):
        """Imprime o estado do loop do vídeo."""
        print("Loop habilitado:", self.loop_video.get())

    def log(self, message):
        """Adiciona mensagens à fila de logs para exibição no terminal."""
        self.log_queue.put(message)

    def process_log_queue(self):
        """Processa a fila de logs e atualiza o widget de terminal."""
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
        """Atualiza a exibição do vídeo na interface a partir da fila de frames."""
        try:
            while True:
                imgtk = self.video_queue.get_nowait()
                self.video_label.configure(image=imgtk)
                self.video_label.imgtk = imgtk
        except queue.Empty:
            pass
        self.after(30, self.update_video)

    # ========================================================
    # ## 4. Métodos de Seleção e Execução
    # ========================================================

    def select_model(self):
        """Permite ao usuário selecionar o modelo YOLO (.pt)."""
        path = filedialog.askopenfilename(title="Selecione o Modelo YOLO",
                                          filetypes=[("Arquivos .pt", "*.pt")])
        if path:
            self.model_path = path
            self.model_ok_label.config(text="✔")
            self.log(f"Modelo selecionado: {os.path.basename(path)}")

    def select_video(self):
        """Permite ao usuário selecionar um vídeo para processamento."""
        path = filedialog.askopenfilename(title="Selecione o Vídeo",
                                          filetypes=[("Vídeos", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.video_ok_label.config(text="✔")
            self.log(f"Vídeo selecionado: {os.path.basename(path)}")

    def run_detection(self):
        """Inicia o processo de detecção de objetos."""
        if not self.model_path or not self.video_path:
            messagebox.showerror("Erro", "Selecione o modelo e o vídeo primeiro.")
            return
        self.log("Iniciando detecção de objetos...")
        threading.Thread(target=self.detect_objects, daemon=True).start()

    def detect_objects(self):
        """
        Realiza a detecção de objetos utilizando o modelo YOLO.
        Atualiza a interface com os resultados e registra os eventos no MongoDB.
        """
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_path)
        objetos = {}
        next_id = 0
        tolerancia = 60
        max_lost = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                if self.loop_video.get():
                    cap.release()
                    cap = cv2.VideoCapture(self.video_path)
                    continue
                else:
                    break

            h_frame, w_frame = frame.shape[:2]
            linha_meio = h_frame // 2
            results = model.predict(frame, verbose=False)
            deteccoes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    classe = "Aprovado" if cls == 0 else "Reprovado"
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    deteccoes.append(((cx, cy), classe))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, classe, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                    percent = ((cy - linha_meio) / h_frame) * 100
                    percent_int = int(round(percent))
                    cv2.putText(frame, f"{percent_int}%", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            for obj in objetos.values():
                obj["prev_pos"] = obj["pos"]
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
            for obj_id in list(objetos.keys()):
                if obj_id not in ids_atualizados:
                    objetos[obj_id]["lost"] += 1
                    if objetos[obj_id]["lost"] > max_lost:
                        del objetos[obj_id]
            for obj in objetos.values():
                y_prev = obj["prev_pos"][1]
                y_atual = obj["pos"][1]
                percent_prev = ((y_prev - linha_meio) / h_frame) * 100
                percent_atual = ((y_atual - linha_meio) / h_frame) * 100
                percent_atual_int = int(round(percent_atual))
                print(f"Objeto {obj['classe']} - Prev: {percent_prev:.2f}%, Atual: {percent_atual:.2f}%")
                if not obj["contado"] and percent_atual_int == 0:
                    obj["contado"] = True
                    self.piece_count += 1
                    fuso_br = pytz.timezone("America/Sao_Paulo")
                    agora_br = datetime.now(fuso_br)
                    data_hora_str = agora_br.strftime("%d/%m/%Y %H:%M:%S")
                    self.log(f"{obj['classe']}, Placa Sextavada, 1318, {data_hora_str} | Total: {self.piece_count}")
                    self.logs_collection.insert_one({
                        "classe": obj["classe"],
                        "nome_item": "Placa Sextavada",
                        "codigo": "1318",
                        "data_hora": agora_br,
                        "total": self.piece_count
                    })
            cv2.line(frame, (0, linha_meio), (w_frame, linha_meio), (0, 0, 255), 2)
            self.video_frame.update_idletasks()
            w_cont = self.video_frame.winfo_width()
            h_cont = self.video_frame.winfo_height()
            escala = min(w_cont / w_frame, h_cont / h_frame)
            novo_w = int(w_frame * escala)
            novo_h = int(h_frame * escala)
            frame_resized = cv2.resize(frame, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_queue.put(imgtk)
        cap.release()
        self.log("Processamento finalizado.")

    # ========================================================
    # ## 5. Métodos de Interface: Ocultar/Restaurar e Alerta Crítico
    # ========================================================

    def hide_app(self):
        """Oculta a interface principal e exibe uma pequena janela para restaurá-la."""
        self.withdraw()
        self.esc_listener = tk.Toplevel(self)
        self.esc_listener.overrideredirect(True)
        self.esc_listener.geometry("300x50+100+100")
        self.esc_listener.config(bg="#2e2e2e")
        info_label = tk.Label(self.esc_listener, text="Pressione ESC para restaurar o aplicativo",
                              bg="#2e2e2e", fg="white", font=("Arial", 10))
        info_label.pack(expand=True, fill="both")
        self.esc_listener.bind("<Escape>", self.show_app)
        self.esc_listener.focus_force()

    def show_app(self, event=None):
        """Restaura a interface principal."""
        self.deiconify()
        if hasattr(self, "esc_listener"):
            self.esc_listener.destroy()

    def show_critical_alert(self, message):
        """
        Exibe um pop-up de alerta crítico com um ícone PNG.
        - Toca um beep (no Windows).
        - Carrega e exibe o ícone 'alert_icon.png' no pop-up.
        - O pop-up se auto-destroi após 3 segundos.
        """
        # Toca um beep (Windows)
        try:
            import winsound
            winsound.Beep(1000, 300)  # Frequência de 1000Hz por 300ms
        except ImportError:
            self.log("winsound não disponível para alerta sonoro.")
        
        # Cria o pop-up de alerta
        alert_win = tk.Toplevel(self)
        alert_win.title("Alerta Crítico")
        alert_win.geometry("300x100")
        alert_win.configure(bg="red")
        
        # Tenta carregar o ícone PNG para o alerta
        try:
            alert_img = Image.open("alert_icon.png")
            alert_img = alert_img.resize((64, 64), Image.Resampling.LANCZOS)
            alert_photo = ImageTk.PhotoImage(alert_img)
            # Exibe o ícone no pop-up
            icon_label = tk.Label(alert_win, image=alert_photo, bg="red")
            icon_label.image = alert_photo  # Mantém uma referência para não ser coletado pelo garbage collector
            icon_label.pack(pady=5)
        except Exception as e:
            self.log(f"Erro ao carregar 'alert_icon.png': {e}")
        
        # Caso queira exibir também uma mensagem textual, pode-se incluir outro label abaixo do ícone
        # Neste exemplo, optamos por não exibir texto adicional ("tire a pena").
        
        # Auto-destruição do pop-up após 3 segundos
        alert_win.after(3000, alert_win.destroy)

    # ========================================================
    # ## 6. Encerramento da Aplicação
    # ========================================================

    def on_close(self):
        """Interrompe os agentes e encerra a aplicação."""
        self.monitor_agent.stop()
        self.supervisor_agent.stop()
        self.destroy()


if __name__ == "__main__":
    app = VideoApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
