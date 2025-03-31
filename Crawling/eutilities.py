import os
import json
import pandas as pd
from Bio import Entrez, Medline
from Crawling.entrez_config import Entrez

# ✅ 여기에 고정 쿼리를 설정 (None이면 사용자 입력받음)
FIXED_QUERY: str | None = None
# FIXED_QUERY = "diabetes AND 2022[dp]"  # 예시

def get_search_query_and_count(email: str, api_key: str) -> tuple[str, int]:
    """
    PubMed 검색어를 받아 결과 개수를 반환한다.
    FIXED_QUERY가 있으면 그걸 쓰고, 없으면 사용자 입력을 요청한다.

    Args:
        email (str): Entrez API에 사용할 이메일
        api_key (str): NCBI API 키

    Returns:
        tuple: (검색어, 검색 결과 개수)
    """
    Entrez.email = email
    Entrez.api_key = api_key

    # 🔍 query 설정 방식 결정
    if FIXED_QUERY is not None:
        query = FIXED_QUERY
        print(f"📌 고정 쿼리 사용: {query}")
    else:
        query = input("🔍 PubMed에서 검색할 키워드를 입력하세요: ").strip()
        if not query:
            print("❗ 검색어를 입력해야 합니다.")
            exit()

    # 🔍 검색 요청
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    result = Entrez.read(handle)
    handle.close()

    total_count = int(result["Count"])
    return query, total_count

def run_eutilities(query: str | None = None):
    query, total_count = get_search_query_and_count(Entrez.email, Entrez.api_key)
    print(f"\n📄 총 {total_count:,}개의 결과가 검색되었습니다.")

    confirm = input("📥 데이터를 다운로드하시겠습니까? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ 다운로드를 취소했습니다.")
        return

    max_by_site = 10000
    max_count = min(max_by_site, total_count)
    retmax = input(f"몇 개의 논문을 가져올까요? (최대 {max_count}개): ").strip()
    retmax = int(retmax) if retmax.isdigit() else 10
    retmax = min(retmax, max_count)

    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]

    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
    records = list(Medline.parse(fetch_handle))
    fetch_handle.close()

    data = [{
        "pmid": r.get("PMID", ""),
        "title": r.get("TI", ""),
        "abstract": r.get("AB", ""),
        "pub_date": r.get("DP", ""),
        "journal": r.get("JT", ""),
        "authors": ", ".join(r.get("AU", []))
    } for r in records]

    save_dir = os.path.join("Database", "eutilities")
    os.makedirs(save_dir, exist_ok=True)

    filename_base = query.replace(" ", "_")[:30]
    json_path = os.path.join(save_dir, f"{filename_base}.json")
    csv_path = os.path.join(save_dir, f"{filename_base}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    pd.DataFrame(data).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료!\n📁 JSON: {json_path}\n📁 CSV: {csv_path}")