"""
Quick in-notebook verification of app.py using FastAPI's TestClient --
exercises all three endpoints without needing a separately running
server. For the real deployment test, run `uvicorn app:app --reload`
in a terminal.
"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

sample_text = (
    "Rahul Verma's phone number is +91-98765-43210 and his email is "
    "rahul.v@company.in. He works at Infosys and is part of Project "
    "Nightingale. His PAN is ABCDE1234F."
)

print("=== /health ===")
print(client.get("/health").json())

print("\n=== /detect (ONNX only -- watch for fragmented entities) ===")
response = client.post("/detect", json={"text": sample_text})
for ent in response.json()["entities"]:
    print(ent)

print("\n=== /detect/hybrid (Presidio + ONNX -- should be clean) ===")
response = client.post("/detect/hybrid", json={"text": sample_text})
for ent in response.json()["entities"]:
    print(ent)

print("\n=== /detect/ensemble (Presidio + GLiNER, high-accuracy path) ===")
response = client.post("/detect/ensemble", json={"text": sample_text})
for ent in response.json()["entities"]:
    print(ent)
