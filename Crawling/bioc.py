import os
import requests

def run_bioc(pmc_id: str):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc_id}/unicode"
    r = requests.get(url)
    if r.status_code == 200:
        save_dir = os.path.join("Database", "bioc")
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"{pmc_id}_bioc.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)

        print(f"✅ BioC XML 저장 완료: {path}")
    else:
        print(f"❌ 요청 실패 (status code: {r.status_code})")