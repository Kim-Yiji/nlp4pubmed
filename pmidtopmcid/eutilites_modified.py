import os
import json
import pandas as pd
from Bio import Entrez, Medline
from Crawling.entrez_config import Entrez

def get_search_query_and_count(email: str, api_key: str, query:str = None) -> tuple[str, int]:
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
    if query is not None:
        query = query
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
    query, total_count = get_search_query_and_count(Entrez.email, Entrez.api_key, query)
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

    # 🔄 PMID to PMCID 매핑 및 full text 수집
    from tqdm import tqdm
    import requests
    import time

    def convert_pmid_to_pmcid(pmid):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"{pmid}[pmid]",
            "retmode": "json",
            "api_key": Entrez.api_key
        }
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            idlist = data.get("esearchresult", {}).get("idlist", [])
            return idlist[0] if idlist else None
        except Exception as e:
            print(f"❌ PMID {pmid} → 변환 실패: {e}")
            return None

    def fetch_fulltext_by_pmcid(pmcid):
        try:
            handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(handle.read(), "lxml")
            handle.close()
            body = soup.find("body")
            if body:
                paras = body.find_all("p")
                return "\n".join(p.get_text(strip=True) for p in paras)
        except Exception as e:
            print(f"❌ PMCID {pmcid} → 본문 수집 실패: {e}")
        return "❌ Full text not available"

    enriched_data = []
    for item in tqdm(data, desc="🔄 PMID → PMCID → Full Text"):
        pmid = item["pmid"]
        pmcid = convert_pmid_to_pmcid(pmid)
        item["pmcid"] = pmcid if pmcid else "N/A"
        item["full_text"] = fetch_fulltext_by_pmcid(pmcid) if pmcid else "❌ No PMCID"
        enriched_data.append(item)
        time.sleep(0.34)

    # 다시 저장
    enriched_json_path = os.path.join(save_dir, f"{filename_base}_full.json")
    enriched_csv_path = os.path.join(save_dir, f"{filename_base}_full.csv")

    with open(enriched_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=2)

    pd.DataFrame(enriched_data).to_csv(enriched_csv_path, index=False, encoding="utf-8-sig")
    print(f"📥 본문 포함 저장 완료!\n📁 JSON: {enriched_json_path}\n📁 CSV: {enriched_csv_path}")