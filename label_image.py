import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import webbrowser
import datetime
from ultralytics import YOLO
import subprocess
from pymongo import MongoClient

# Define o período de validade da versão beta
BETA_DURATION = datetime.timedelta(days=3)

class LabelingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicação de Rotulação de Imagens - Vega Robotics")
        self.geometry("1000x700")

        # Conexão com o MongoDB (ajuste a URI conforme necessário)
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["monitoramento"]  # Nome do banco
            self.collection = self.db["deteccoes"]          # Nome da coleção
        except Exception as e:
            messagebox.showerror("Erro MongoDB", f"Não foi possível conectar ao MongoDB: {e}")
            self.mongo_client = None

        # Variáveis de configuração
        self.photo_dir = None           # Diretório das fotos (input)
        self.test_video_path = None     # PATH do vídeo teste
        self.test_model_path = None     # PATH da rede neural para teste (opcional)
        self.classes = []               # Lista das classes definidas
        self.photo_list = []            # Lista dos caminhos das fotos
        self.current_index = 0          # Índice da foto atual

        # Dados da imagem atual
        self.current_image = None       # Objeto PIL.Image
        self.photo_image = None         # Para exibição no Canvas
        self.img_width = self.img_height = 0

        # Armazenamento interno das anotações (para rotulagem)
        self.labeled_annotations = {}   # Mapeia <nome_base> -> conteúdo do TXT YOLO
        self.labeled_image_paths = []   # Lista dos caminhos das imagens rotuladas

        # Variáveis para desenho (bounding box)
        self.drawing = False
        self.start_x = self.start_y = 0
        self.current_rect = None

        # Caminho do modelo treinado (best.pt) para uso futuro
        self.best_model_path = None

        # Arquivo data.yaml gerado durante o treinamento (usado na validação)
        self.data_yaml_path = None

        # Tela inicial
        self.create_config_frame()

    def add_footer(self, parent):
        """Adiciona um rodapé com o nome da empresa e informações do autor."""
        footer = tk.Frame(parent, bd=1, relief="sunken")
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(footer, text="Vega Robotics", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=5)
        autor = tk.Label(footer, text="saber mais sobre autor: Rafael Maciel", fg="blue", cursor="hand2", font=("Helvetica", 9, "italic"))
        autor.pack(side=tk.RIGHT, padx=5)
        autor.bind("<Button-1>", lambda e: webbrowser.open("https://www.linkedin.com/in/rafael-s-maciel/"))

    def check_beta_validity(self):
        """Verifica a data de início da versão beta e retorna o tempo restante."""
        beta_file = "beta_start.txt"
        if os.path.exists(beta_file):
            with open(beta_file, "r") as f:
                start_str = f.read().strip()
            try:
                beta_start = datetime.datetime.fromisoformat(start_str)
            except Exception:
                beta_start = datetime.datetime.now()
                with open(beta_file, "w") as f:
                    f.write(beta_start.isoformat())
        else:
            beta_start = datetime.datetime.now()
            with open(beta_file, "w") as f:
                f.write(beta_start.isoformat())
        beta_end = beta_start + BETA_DURATION
        now = datetime.datetime.now()
        remaining = beta_end - now
        if remaining.total_seconds() <= 0:
            messagebox.showerror("Beta Expirada", "Esta versão BETA expirou. Por favor, atualize para a versão estável.")
            self.destroy()
        return remaining

    def update_beta_counter(self):
        """Atualiza o contador da versão beta a cada segundo."""
        remaining = self.check_beta_validity()
        days = remaining.days
        seconds = remaining.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        text = f"Versão BETA - válida por 3 dias. Tempo restante: {days}d {hours}h {minutes}m {secs}s"
        self.beta_label.config(text=text)
        self.after(1000, self.update_beta_counter)

    # -----------------------------------------
    # 1) Tela de Configuração Inicial
    # -----------------------------------------
    def create_config_frame(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.config_frame = tk.Frame(self)
        self.config_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(self.config_frame, text="Configuração do Projeto - Vega Robotics", font=("Helvetica", 16, "bold")).pack(pady=10)
        tk.Button(self.config_frame, text="Selecionar PATH das Fotos", command=self.select_photo_dir).pack(pady=5)
        self.lbl_photo_dir = tk.Label(self.config_frame, text="Nenhum diretório selecionado")
        self.lbl_photo_dir.pack(pady=2)
        tk.Button(self.config_frame, text="Selecionar PATH do Vídeo Teste", command=self.select_test_video).pack(pady=5)
        self.lbl_test_video = tk.Label(self.config_frame, text="Nenhum vídeo selecionado")
        self.lbl_test_video.pack(pady=2)
        tk.Button(self.config_frame, text="Selecionar PATH da Rede Neural para Teste", command=self.select_test_model).pack(pady=5)
        self.lbl_test_model = tk.Label(self.config_frame, text="Nenhuma rede neural selecionada")
        self.lbl_test_model.pack(pady=2)
        tk.Label(self.config_frame, text="Digite os nomes das classes (separados por vírgula):").pack(pady=5)
        self.classes_entry = tk.Entry(self.config_frame, width=50)
        self.classes_entry.pack(pady=5)
        tk.Button(self.config_frame, text="Iniciar Rotulação", command=self.start_labeling).pack(pady=20)
        self.beta_label = tk.Label(self.config_frame, text="", font=("Helvetica", 10, "italic"), fg="red")
        self.beta_label.pack(pady=5)
        self.update_beta_counter()
        self.add_footer(self.config_frame)

    def select_photo_dir(self):
        dir_path = filedialog.askdirectory(title="Selecione o diretório das fotos")
        if dir_path:
            self.photo_dir = dir_path
            self.lbl_photo_dir.config(text=self.photo_dir)

    def select_test_video(self):
        file_path = filedialog.askopenfilename(title="Selecione o vídeo teste", filetypes=[("Arquivos de Vídeo", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.test_video_path = file_path
            self.lbl_test_video.config(text=self.test_video_path)

    def select_test_model(self):
        file_path = filedialog.askopenfilename(title="Selecione a rede neural para teste", filetypes=[("Arquivos de Peso", "*.pt")])
        if file_path:
            self.test_model_path = file_path
            self.lbl_test_model.config(text=self.test_model_path)

    def start_labeling(self):
        if not self.photo_dir:
            messagebox.showerror("Erro", "Selecione o diretório das fotos.")
            return
        classes_str = self.classes_entry.get().strip()
        if not classes_str:
            messagebox.showerror("Erro", "Informe pelo menos uma classe.")
            return
        self.classes = [c.strip() for c in classes_str.split(',') if c.strip()]
        if len(self.classes) == 0:
            messagebox.showerror("Erro", "Nenhuma classe válida foi inserida.")
            return
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.photo_list = [os.path.join(self.photo_dir, f) for f in os.listdir(self.photo_dir) if f.lower().endswith(valid_ext)]
        if not self.photo_list:
            messagebox.showerror("Erro", "Nenhuma imagem encontrada no diretório selecionado.")
            return
        if len(self.photo_list) < 3:
            messagebox.showwarning("Aviso", "Poucas imagens para treinamento. Recomenda-se pelo menos 3 para uma divisão mínima.")
        self.config_frame.destroy()
        self.create_labeling_frame()
        self.load_next_photo()

    # -----------------------------------------
    # 2) Tela de Rotulagem
    # -----------------------------------------
    def create_labeling_frame(self):
        self.label_frame = tk.Frame(self)
        self.label_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame = tk.Frame(self.label_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=800, height=600, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar = tk.Scrollbar(self.label_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.pack(fill=tk.X)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        control_frame = tk.Frame(self.label_frame)
        control_frame.pack(pady=5)
        tk.Button(control_frame, text="Salvar e Próxima Foto", command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Treinar Modelo", command=self.open_train_config_window).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Voltar ao Menu Inicial", command=self.return_to_main_menu).pack(side=tk.LEFT, padx=5)
        self.add_footer(self.label_frame)

    def load_next_photo(self):
        if self.current_index >= len(self.photo_list):
            messagebox.showinfo("Finalizado", "Não há mais fotos para rotular.")
            return
        photo_path = self.photo_list[self.current_index]
        self.current_index += 1
        self.annotations = []  # Reseta as anotações para a foto atual
        self.current_image = Image.open(photo_path)
        self.img_width, self.img_height = self.current_image.size
        canvas_width, canvas_height = 800, 600
        region_width = max(self.img_width, canvas_width)
        region_height = max(self.img_height, canvas_height)
        self.canvas.config(scrollregion=(0, 0, region_width, region_height))
        offset_x = (canvas_width - self.img_width) // 2 if self.img_width < canvas_width else 0
        offset_y = (canvas_height - self.img_height) // 2 if self.img_height < canvas_height else 0
        self.photo_image = ImageTk.PhotoImage(self.current_image)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=self.photo_image, anchor=tk.NW)

    def on_button_press(self, event):
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_move_press(self, event):
        if not self.drawing:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.canvas.delete(self.current_rect)
            messagebox.showinfo("Aviso", "Caixa muito pequena, descartada.")
            return
        self.canvas.delete(self.current_rect)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)
        self.open_class_selector(x1, y1, x2, y2)

    def open_class_selector(self, x1, y1, x2, y2):
        top = tk.Toplevel(self)
        top.title("Selecione a Classe - Vega Robotics")
        tk.Label(top, text="Selecione a classe:").pack(pady=5)
        class_var = tk.StringVar()
        combo = ttk.Combobox(top, textvariable=class_var, values=self.classes, state="readonly")
        combo.pack(pady=5)
        combo.current(0)
        def confirm():
            classe = class_var.get()
            if not classe:
                messagebox.showerror("Erro", "Selecione uma classe.")
                return
            self.annotations.append((x1, y1, x2 - x1, y2 - y1, classe))
            top.destroy()
        tk.Button(top, text="OK", command=confirm).pack(pady=5)
        self.add_footer(top)

    def save_annotation(self):
        if not self.current_image:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return
        if not self.annotations:
            messagebox.showerror("Erro", "Nenhuma anotação realizada nesta imagem.")
            return
        current_photo_path = self.photo_list[self.current_index - 1]
        base_name = os.path.splitext(os.path.basename(current_photo_path))[0]
        w, h = self.img_width, self.img_height
        content = ""
        for (x, y, box_w, box_h, classe) in self.annotations:
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            width_norm = box_w / w
            height_norm = box_h / h
            class_id = self.classes.index(classe) if classe in self.classes else 0
            content += f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        self.labeled_annotations[base_name] = content
        self.labeled_image_paths.append(current_photo_path)
        messagebox.showinfo("Salvo", f"Anotações para {base_name} salvas internamente.")

    def save_and_next(self):
        self.save_annotation()
        self.load_next_photo()

    def return_to_main_menu(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.current_index = 0
        self.create_config_frame()

    # -----------------------------------------
    # 3) Tela de Treinamento
    # -----------------------------------------
    def open_train_config_window(self):
        self.train_win = tk.Toplevel(self)
        self.train_win.title("Configurar Treinamento - Vega Robotics")
        self.train_win.geometry("400x350")
        tk.Label(self.train_win, text="Parâmetros de Treinamento", font=("Helvetica", 16, "bold")).pack(pady=10)
        tk.Label(self.train_win, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self.train_win, width=10)
        self.epochs_entry.insert(0, "150")
        self.epochs_entry.pack(pady=5)
        tk.Label(self.train_win, text="Tamanho da Imagem (imgsz):").pack()
        self.imgsz_entry = tk.Entry(self.train_win, width=10)
        self.imgsz_entry.insert(0, "640")
        self.imgsz_entry.pack(pady=5)
        tk.Label(self.train_win, text="Workers:").pack()
        self.workers_entry = tk.Entry(self.train_win, width=10)
        self.workers_entry.insert(0, "2")
        self.workers_entry.pack(pady=5)
        self.train_status_label = tk.Label(self.train_win, text="Aguardando início do treinamento...")
        self.train_status_label.pack(pady=5)
        tk.Button(self.train_win, text="Iniciar Treinamento", command=self.start_training).pack(pady=10)
        self.progress = ttk.Progressbar(self.train_win, mode="indeterminate")
        self.progress.pack(pady=10, fill=tk.X, padx=20)
        self.add_footer(self.train_win)

    def start_training(self):
        try:
            self.train_epochs = int(self.epochs_entry.get())
            self.train_imgsz = int(self.imgsz_entry.get())
            self.train_workers = int(self.workers_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Informe valores válidos para os parâmetros de treinamento.")
            return
        if not self.labeled_annotations:
            messagebox.showerror("Erro", "Nenhuma imagem rotulada para treinar.")
            return
        self.train_status_label.config(text="Treinamento em andamento...")
        self.progress.start(10)
        self.train_win.update_idletasks()
        t = threading.Thread(target=self.train_model, args=(self.train_epochs, self.train_imgsz, self.train_workers))
        t.start()

    # -----------------------------------------
    # 4) Treinamento e Exibição dos Gráficos
    # -----------------------------------------
    def train_model(self, epochs, imgsz, workers):
        all_images = list(self.labeled_annotations.keys())
        total = len(all_images)
        if total == 0:
            self.after(0, lambda: messagebox.showerror("Erro", "Nenhuma imagem rotulada para treinamento."))
            return
        n_train = total // 3
        n_val = total // 3
        n_test = total - n_train - n_val
        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train+n_val]
        test_imgs = all_images[n_train+n_val:]
        output_dir = os.path.join(os.getcwd(), "output")
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        def process_set(image_set, target_dir):
            for base_name in image_set:
                for path in self.labeled_image_paths:
                    if os.path.splitext(os.path.basename(path))[0] == base_name:
                        shutil.copy2(path, target_dir)
                        txt_path = os.path.join(target_dir, base_name + ".txt")
                        with open(txt_path, "w") as f:
                            f.write(self.labeled_annotations[base_name])
                        break
        process_set(train_imgs, train_dir)
        process_set(val_imgs, val_dir)
        process_set(test_imgs, test_dir)
        data_yaml_path = os.path.join(output_dir, "data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write(f"train: {train_dir}\n")
            f.write(f"val: {val_dir}\n")
            f.write(f"test: {test_dir}\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write("names: " + str(self.classes) + "\n")
        self.data_yaml_path = data_yaml_path
        print("Iniciando treinamento com os seguintes parâmetros:")
        print(f"  data: {data_yaml_path}")
        print(f"  epochs: {epochs}, imgsz: {imgsz}, workers: {workers}")
        try:
            modelo = YOLO('yolov8n.pt')
            modelo.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz, workers=workers)
            self.after(0, lambda: messagebox.showinfo("Treinamento", "Treinamento finalizado!"))
            self.after(0, self.plot_metrics)
            exp_dir = self.get_latest_exp_dir()
            if exp_dir:
                self.best_model_path = os.path.join(exp_dir, "weights", "best.pt")
                print("Modelo treinado salvo em:", self.best_model_path)
            else:
                print("Nenhum experimento YOLO encontrado em runs/detect/ ou runs/train/.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro no Treinamento", str(e)))
        finally:
            self.after(0, self.stop_progress)

    def stop_progress(self):
        self.progress.stop()
        if hasattr(self, "train_win"):
            self.train_win.destroy()

    def get_latest_exp_dir(self):
        base_dirs = [
            os.path.join(os.getcwd(), "runs", "detect"),
            os.path.join(os.getcwd(), "runs", "train")
        ]
        exp_dirs = []
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                exp_dirs.extend([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if exp_dirs:
            latest_dir = max(exp_dirs, key=os.path.getmtime)
            return latest_dir
        return None

    def plot_metrics(self):
        exp_dir = self.get_latest_exp_dir()
        if not exp_dir:
            messagebox.showerror("Erro", "Nenhum experimento YOLO encontrado.")
            return
        graph_files = {
            "Confusion Matrix": ("confusion_matrix.png", "Matriz de Confusão: Compara classes reais e previstas."),
            "Recall Curve": ("R_curve.png", "Curva de Recall: Mostra a capacidade do modelo de recuperar instâncias positivas."),
            "F1 Score Curve": ("F1_curve.png", "Curva de F1 Score: Balanceia Precisão e Recall."),
            "Precision Curve": ("P_curve.png", "Curva de Precisão: Indica a proporção de verdadeiros positivos entre as predições positivas.")
        }
        graph_win = tk.Toplevel(self)
        graph_win.title("Gráficos do Treinamento YOLO - Vega Robotics")
        graph_win.state("zoomed")
        canvas = tk.Canvas(graph_win, bg="white")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(graph_win, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        for title, (filename, explanation) in graph_files.items():
            file_path = os.path.join(exp_dir, filename)
            tk.Label(frame, text=title, font=("Helvetica", 14, "bold")).pack(pady=5)
            if os.path.exists(file_path):
                img = Image.open(file_path)
                max_width, max_height = 800, 600
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl_img = tk.Label(frame, image=photo, borderwidth=2, relief="groove")
                lbl_img.image = photo
                lbl_img.pack(pady=5)
                tk.Label(frame, text=explanation, wraplength=800, justify="left").pack(pady=5)
            else:
                tk.Label(frame, text=f"Arquivo não encontrado: {filename}").pack(pady=5)
        metrics_text = ""
        try:
            modelo_eval = YOLO(self.best_model_path)
            val_results = modelo_eval.val(data=self.data_yaml_path, verbose=False)
            cm = val_results[0].confusion_matrix
            VP = cm[1, 1]
            VN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]
            total_samples = cm.sum()
            accuracy = (VP + VN) / total_samples if total_samples > 0 else 0
            metrics_text = (
                f"Cálculo da Taxa de Acurácia:\n"
                f"Acurácia = (VP + VN) / Total de Amostras = ({VP} + {VN}) / {total_samples} = {accuracy:.2f}\n\n"
                f"Extração dos Falsos Positivos (FP):\n"
                f"FP = {FP} (casos em que a classe 0 foi predita como classe 1)\n"
                f"Falsos Negativos (FN) = {FN} (casos em que a classe 1 foi predita como classe 0)\n\n"
                f"Cálculo do R²: Não se aplica diretamente a problemas de classificação."
            )
        except Exception:
            metrics_text = ""
        if metrics_text:
            metrics_frame = tk.Frame(graph_win, bd=1, relief="solid")
            metrics_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
            tk.Label(metrics_frame, text="Métricas Calculadas", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=5, pady=2)
            tk.Label(metrics_frame, text=metrics_text, font=("Helvetica", 10), justify="left").pack(anchor="w", padx=5, pady=2)
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        button_frame = tk.Frame(graph_win)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        tk.Button(button_frame, text="Voltar ao Menu Inicial", command=lambda: [graph_win.destroy(), self.return_to_main_menu()]).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Rodar Vídeo Teste", command=self.run_test_video).pack(side=tk.RIGHT, padx=10)
        tk.Label(button_frame, text="Sugestão: Utilize o best.pt num vídeo teste.", font=("Helvetica", 10, "italic")).pack(pady=5)
        self.add_footer(graph_win)

    # -----------------------------------------
    # 5) Seleção da Rede Neural e Execução do Vídeo Teste
    # -----------------------------------------
    def open_model_selector(self):
        selector = tk.Toplevel(self)
        selector.title("Selecione a Rede Neural para Teste - Vega Robotics")
        var = tk.StringVar()
        options = {}
        if self.best_model_path:
            options["Treinada (best.pt)"] = self.best_model_path
        if self.test_model_path:
            options["Selecionada no Menu"] = self.test_model_path
        options["Selecionar outro arquivo..."] = "other"
        tk.Label(selector, text="Selecione a rede neural a ser utilizada:").pack(pady=5)
        for key, value in options.items():
            texto = f"{key}: {value}" if value != "other" else key
            tk.Radiobutton(selector, text=texto, variable=var, value=value).pack(anchor="w", padx=20)
        if options:
            if self.best_model_path:
                var.set(self.best_model_path)
            else:
                var.set(list(options.values())[0])
        chosen_model = {}
        def confirm_selection():
            sel = var.get()
            if sel == "other":
                file_path = filedialog.askopenfilename(title="Selecione a rede neural para teste", filetypes=[("Arquivos de Peso", "*.pt")])
                if file_path:
                    chosen_model["path"] = file_path
                    selector.destroy()
            else:
                chosen_model["path"] = sel
                selector.destroy()
        tk.Button(selector, text="Confirmar", command=confirm_selection).pack(pady=10)
        selector.grab_set()
        self.wait_window(selector)
        return chosen_model.get("path", None)

    def run_test_video(self):
        selected_model = self.open_model_selector()
        if not selected_model:
            messagebox.showerror("Erro", "Nenhuma rede neural selecionada.")
            return
        if not self.test_video_path:
            messagebox.showerror("Erro", "Selecione um vídeo teste no menu inicial.")
            return
        t = threading.Thread(target=self.video_test_thread, args=(selected_model,))
        t.start()

    def video_test_thread(self, model_path):
        import cv2
        modelo = YOLO(model_path)
        video = cv2.VideoCapture(self.test_video_path)
        nomes_classes = self.classes if self.classes else ['Classe0']
        classes_ocultar = []  # Pode ser personalizada
        while True:
            check, img = video.read()
            if not check:
                break
            resultado = modelo.predict(img, verbose=False)
            for obj in resultado:
                for item in obj.boxes:
                    x1, y1, x2, y2 = item.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(item.cls[0])
                    nomeClasse = nomes_classes[cls] if cls < len(nomes_classes) else "Classe"
                    conf = round(float(item.conf[0]), 2)
                    print(f"Detecção: {nomeClasse}, Confiança: {conf}%")
                    # Persistência no MongoDB
                    if self.mongo_client:
                        from datetime import datetime
                        doc = {
                            "timestamp": datetime.now(),
                            "classe": nomeClasse,
                            "confianca": conf,
                            "bounding_box": [x1, y1, x2, y2],
                            "video": self.test_video_path
                        }
                        self.collection.insert_one(doc)
                    # Chama outro script se a classe for "Alerta"
                    if nomeClasse == "Alerta" and conf > 0.5:
                        subprocess.Popen(["python", "outro_script.py", "--classe", nomeClasse])
                    if nomeClasse in classes_ocultar or (nomeClasse == 'Tampa' and conf <= 0.30):
                        continue
                    cv2.putText(img, nomeClasse, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,0), thickness=2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 2)
            cv2.imshow('Vídeo Teste', img)
            if cv2.waitKey(1) == 27:
                break
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LabelingApp()
    app.mainloop()
