#pip install biopython

from Bio import Entrez
import time

Entrez.email = "" # <- 본인 이메일

# 키워드로 PMC 논문 검색
def search_pmc(keyword, max_results=100):
    handle = Entrez.esearch(db="pmc", term=keyword, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

# 논문 XML 다운로드
def fetch_pmc(pmc_id):
    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
    xml = handle.read()
    handle.close()
    return xml

search_keyword = "Drug Interaction"  # <-- 검색할 키워드
max_download_articles = 10           # <-- 다운로드할 논문 수 (필수입력)

# 예시 사용법
pmc_ids = search_pmc("Drug Interaction", max_results=10)

for pmc_id in pmc_ids:
    xml_data = fetch_pmc(pmc_id).decode('utf-8')
    with open(f"{pmc_id}.xml", "w", encoding="utf-8") as f:
        f.write(xml_data)
    time.sleep(0.5)  # 과도한 요청 방지