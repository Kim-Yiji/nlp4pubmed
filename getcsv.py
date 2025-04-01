#pip install biopython
#pip install pandas
#pip install lxml

from Bio import Entrez
import pandas as pd
import time

# 본인 이메일 필수 입력!
Entrez.email = "your_email@example.com"  # <-- 필수 수정!

def search_pmc(keyword, max_results=10):
    query = f'("{keyword}"[Title] OR "{keyword}"[Abstract])'
    handle = Entrez.esearch(db="pmc", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

def fetch_pmc_metadata(pmc_id):
    handle = Entrez.efetch(db="pmc", id=pmc_id, retmode="xml")
    xml_data = handle.read()
    handle.close()
    return xml_data

def parse_pmc_xml(xml_data):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(xml_data, 'lxml')

    title_tag = soup.find('article-title')
    title = title_tag.get_text(strip=True) if title_tag else ""

    abstract_tags = soup.find_all('abstract')
    abstracts = ' '.join(ab.get_text(strip=True) for ab in abstract_tags)

    return title, abstracts

# 사용자 입력 부분
search_keyword = "drug interaction"  # <-- 검색 키워드 입력!
max_articles = 10                         # <-- 논문 최대 개수 입력!

pmc_ids = search_pmc(search_keyword, max_results=max_articles)

data = []
for pmc_id in pmc_ids:
    xml_data = fetch_pmc_metadata(pmc_id)
    title, abstract = parse_pmc_xml(xml_data)
    data.append({
        'PMC_ID': pmc_id,
        'Title': title,
        'Abstract': abstract
    })
    print(f"{pmc_id} 완료")
    time.sleep(0.5)

# CSV로 저장하기
df = pd.DataFrame(data)
df.to_csv("pmc_articles.csv", index=False, encoding='utf-8')

print("CSV 저장 완료: pmc_articles.csv")