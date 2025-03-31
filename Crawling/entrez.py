import json
import os
import pandas as pd
from Bio import Entrez, Medline

# 📌 config 불러오기
with open("config.json", "r") as f:
    config = json.load(f)

Entrez.email = config["email"]

# 🔍 검색어 설정
query = "cancer AND 2023[dp]" # 최대 10000건이라 년도로 필터링해서 가져올것

# 🔍 논문 검색
search_handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
search_results = Entrez.read(search_handle)
search_handle.close()

id_list = search_results["IdList"]

# 📄 논문 상세 정보 가져오기 (Medline 형식은 구조화되어 있어 편함!)
fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
records = Medline.parse(fetch_handle)
records = list(records)
fetch_handle.close()

# 📊 원하는 필드 추출해서 정리
data = []
for record in records:
    data.append({
        "pmid": record.get("PMID", ""),
        "title": record.get("TI", ""),
        "abstract": record.get("AB", ""),
        "pub_date": record.get("DP", ""),
        "journal": record.get("JT", ""),
        "authors": ", ".join(record.get("AU", []))  # 리스트 → 문자열
    })

# 📁 저장 폴더 만들기
os.makedirs("Database", exist_ok=True)

# ✅ 저장: JSON
with open("Database/pubmed_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# ✅ 저장: CSV
df = pd.DataFrame(data)
df.to_csv("Database/pubmed_data.csv", index=False, encoding="utf-8-sig")

print("✔️ PubMed 데이터 저장 완료!")