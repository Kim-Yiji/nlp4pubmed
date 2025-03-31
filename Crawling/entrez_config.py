import json
from Bio import Entrez

# 📌 config 불러오기 및 설정
with open("config.json", "r") as f:
    config = json.load(f)

Entrez.email = config["email"]
Entrez.api_key = config["api_key"]