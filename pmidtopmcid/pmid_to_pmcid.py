from dotenv import load_dotenv
import requests
import time
from tqdm import tqdm
import os
import json

# 환경변수에서 EMAIL과 API_KEY 불러오기
load_dotenv()
EMAIL = os.getenv("EMAIL")
API_KEY = os.getenv("API_KEY")

def pmid_to_pmcid(pmid):
    """Convert a single PMID to PMCID using NCBI's E-utilities API."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": f"{pmid}[pmid]",
        "retmode": "json",
        "email": EMAIL,
        "api_key": API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            pmcid_list = data["esearchresult"]["idlist"]
            if pmcid_list:
                return f"PMC{pmcid_list[0]}"  # Add PMC prefix
        return None
    except Exception as e:
        print(f"Error converting PMID {pmid}: {str(e)}")
        return None

def main():
    output_file = "pmid_to_pmcid.json"
    failed_file = "pmid_failed.txt"
    
    # Read PMIDs from file
    with open("pmid_list_half.txt", "r") as f:
        pmids = [line.strip() for line in f if line.strip()]
    
    # Get already processed PMIDs if file exists
    processed_pmids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
                processed_pmids = set(data.keys())
            except json.JSONDecodeError:
                # If file is empty or corrupted, start fresh
                pass
    
    # Filter out already processed PMIDs
    pmids = [pmid for pmid in pmids if pmid not in processed_pmids]
    total_pmids = len(pmids)
    
    # Initialize or load existing data
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    
    # Convert PMIDs to PMCIDs and update data
    for i, pmid in enumerate(tqdm(pmids, desc="Converting PMIDs to PMCIDs", total=total_pmids), 1):
        pmcid = pmid_to_pmcid(pmid)
        if pmcid:
            data[pmid] = pmcid
            # Save after each successful conversion
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(failed_file, "a") as f:
                f.write(f"{pmid}\n")
        time.sleep(0.1)  # Respect NCBI's rate limit

    print(f"✅ 변환 완료: {len(data)}개 성공, {total_pmids - len(data)}개 실패")
    print(f"📄 실패한 PMID는 '{failed_file}'에 기록됨")

if __name__ == "__main__":
    main() 