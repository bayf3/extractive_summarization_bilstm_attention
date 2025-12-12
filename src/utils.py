# src/utils.py
import json
import torch
import os

def save_json(obj, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
