import json
import os
import pandas as pd
from Bio import Entrez, Medline
from entrez_search import get_search_query_and_count  # ✅ 모듈 import

# 📌 config 불러오기
with open("config.json", "r") as f:
    config = json.load(f)

Entrez.email = config["email"]
Entrez.api_key = config["api_key"]

# ✅ 여기서 query를 None 또는 문자열로 지정
query = None
# query = "diabetes AND 2022[dp]"  # <- 이걸로 고정 검색하고 싶다면 이 줄만 살리면 됨

# 🔍 검색어 처리
query, total_count = get_search_query_and_count(config["email"], config["api_key"], query)
print(f"\n📄 총 {total_count:,}개의 결과가 검색되었습니다.")

# ✅ 다운로드 여부 확인
confirm = input("📥 데이터를 다운로드하시겠습니까? (y/n): ").strip().lower()
if confirm != "y":
    print("❌ 다운로드를 취소했습니다.")
    exit()

# ✅ 몇 개 가져올지
max_by_site = 10000
max_count = min(max_by_site, total_count)
retmax = input(f"몇 개의 논문을 가져올까요? (최대 {max_count}개): ").strip()
retmax = int(retmax) if retmax.isdigit() else 10
retmax = min(retmax, max_count)

# 🔍 다시 ID 검색
search_handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
search_results = Entrez.read(search_handle)
search_handle.close()
id_list = search_results["IdList"]

# 📄 논문 가져오기
fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
records = Medline.parse(fetch_handle)
records = list(records)
fetch_handle.close()

# 📊 정리
data = []
for record in records:
    data.append({
        "pmid": record.get("PMID", ""),
        "title": record.get("TI", ""),
        "abstract": record.get("AB", ""),
        "pub_date": record.get("DP", ""),
        "journal": record.get("JT", ""),
        "authors": ", ".join(record.get("AU", []))
    })

# 💾 저장
os.makedirs("Database", exist_ok=True)
filename_base = query.replace(" ", "_")[:30]
json_path = f"Database/{filename_base}.json"
csv_path = f"Database/{filename_base}.csv"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(data)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 저장 완료!\n📁 JSON: {json_path}\n📁 CSV: {csv_path}")