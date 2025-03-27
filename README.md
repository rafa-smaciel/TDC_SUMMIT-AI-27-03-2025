# TDC - Monitoramento Industrial com Visão Computacional e Sistemas Multiagentes (MAS)

Projeto apresentado no **TDC 2025** demonstrando a integração entre **Visão Computacional**, **Inteligência Artificial**, **MongoDB**, **FastAPI** e **Sistemas Multiagentes (MAS)** para otimização de processos industriais.

## 🌐 Visão Geral
Aplicativo completo para:

- Detecção de peças em vídeo usando YOLO
- Interface em tempo real com Tkinter
- Registro automatizado dos eventos no MongoDB
- API REST com FastAPI para consulta e integração
- Arquitetura pronta para expansão com agentes inteligentes (MAS)

---

## 📁 Estrutura do Projeto

```
C:\TDC\TDC - 27-03-25
|
├── app_visual/           # Aplicativo com interface Tkinter + YOLO
│   └── main.py
|
├── api/                  # API REST com FastAPI
│   └── main.py
|
├── database/             # Scripts de exportação e manipulação de dados
│   └── export_to_json.py
|
├── requirements.txt      # Lista de dependências do projeto
|
├── README.md             # Este arquivo
|
└── venv/ (ou tdc/)       # Ambiente virtual (ignorado pelo Git)
```

---

## 🚀 Como Executar

### 1. Ativar o ambiente virtual
```bash
cd "C:\TDC\TDC - 27-03-25"
venv\Scripts\activate
```

### 2. Instalar as dependências (se ainda não instaladas)
```bash
pip install -r requirements.txt
```

### 3. Executar o app de Visão Computacional
```bash
cd app_visual
python main.py
```

### 4. Executar a API REST (opcional)
```bash
cd ..\api
uvicorn main:app --reload
```

Acesse via navegador:
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Tecnologias Utilizadas
- Python 3.8+
- OpenCV
- YOLO (Ultralytics)
- MongoDB / PyMongo
- FastAPI
- Tkinter
- Pytz / datetime

---

## 🧰 Público Alvo
Engenheiros de software, cientistas de dados, profissionais de automação industrial, desenvolvedores e interessados em inteligência artificial aplicada à Indústria 4.0.

---

## 🤖 Contato / Créditos
**Apresentação:** TDC 2025 - Workshop de Visão Computacional e Sistemas Multiagentes

Este projeto foi desenvolvido como parte de uma demonstração prática para integração de IA e automação industrial.

> Sinta-se à vontade para clonar, adaptar e colaborar!!