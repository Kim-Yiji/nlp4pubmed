import json
import os
import pandas as pd
from Bio import Entrez, Medline
from entrez_search import get_search_query_and_count  # ✅ 모듈 import

# 📌 config 불러오기
with open("config.json", "r") as f:
    config = json.load(f)

Entrez.email = config["email"]

# ✅ 검색어 입력 & 검색 결과 개수 확인
query, total_count = get_search_query_and_count(config["email"])
print(f"\n📄 총 {total_count:,}개의 결과가 검색되었습니다.")

# ✅ 다운로드 여부 확인
confirm = input("📥 데이터를 다운로드하시겠습니까? (y/n): ").strip().lower()
if confirm != "y":
    print("❌ 다운로드를 취소했습니다.")
    exit()

# ✅ 가져올 논문 개수 선택
max_count = min(10000, total_count)
retmax = input(f"몇 개의 논문을 가져올까요? (최대 {max_count}개): ").strip() # 웬만하면 년도로 filter 해서 가져오기
retmax = int(retmax) if retmax.isdigit() else 10000
retmax = min(retmax, max_count)

# 🔍 PubMed ID 재검색
search_handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
search_results = Entrez.read(search_handle)
search_handle.close()

id_list = search_results["IdList"]

# 📄 논문 상세정보 가져오기
fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
records = Medline.parse(fetch_handle)
records = list(records)
fetch_handle.close()

# 📊 데이터 정리
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