# agents.py
import threading
import time
from datetime import datetime
from pymongo import MongoClient

# Classe base para agentes
class BaseAgent(threading.Thread):
    def __init__(self, name):
        super().__init__(daemon=True)
        self.name = name
        self.running = False

    def run(self):
        self.running = True
        print(f"[{self.name}] Iniciado.")
        while self.running:
            self.listen()
            time.sleep(1)

    def listen(self):
        raise NotImplementedError("Subclasse deve implementar listen().")

    def stop(self):
        self.running = False


# Agente Supervisor (reage a eventos reprovados)
class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Supervisor")

    def process_event(self, event):
        if event.get("classe") == "Reprovado":
            print(f"[Supervisor] ⚠️ Evento CRÍTICO detectado: {event}")
            # Aqui você pode acionar alertas, logs, notificações, etc.
        else:
            print(f"[Supervisor] ✅ Evento aprovado ignorado.")

    def listen(self):
        pass  # Age apenas via callback


# Agente que monitora o banco de dados
class EventMonitorAgent(BaseAgent):
    def __init__(self, callback):
        super().__init__("Monitor")
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
                print(f"[Monitor] Novo evento detectado: {e}")
                if self.callback:
                    self.callback(e)


# Execução dos agentes
if __name__ == "__main__":
    supervisor = SupervisorAgent()
    monitor = EventMonitorAgent(callback=supervisor.process_event)

    print("Iniciando agentes...")
    supervisor.start()
    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Encerrando...")
        monitor.stop()
        supervisor.stop()
        monitor.join()
        supervisor.join()
        print("Agentes finalizados.")
