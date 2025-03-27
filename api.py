from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TDC Workshop API")

# Configuração de CORS para permitir acesso de qualquer origem (ajuste conforme necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conecta ao MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["tdc_workshop"]
collection = db["event_logs"]

def convert_id(event):
    event["_id"] = str(event["_id"])
    return event

@app.get("/eventos")
def get_eventos():
    """
    Retorna todos os eventos.
    """
    eventos = list(collection.find())
    eventos = [convert_id(e) for e in eventos]
    return eventos

@app.get("/eventos/reprovados")
def get_reprovados():
    """
    Retorna apenas os eventos com classe 'Reprovado'.
    """
    eventos = list(collection.find({"classe": "Reprovado"}))
    eventos = [convert_id(e) for e in eventos]
    return eventos

@app.get("/eventos/aprovados")
def get_aprovados():
    """
    Retorna apenas os eventos com classe 'Aprovado'.
    """
    eventos = list(collection.find({"classe": "Aprovado"}))
    eventos = [convert_id(e) for e in eventos]
    return eventos
