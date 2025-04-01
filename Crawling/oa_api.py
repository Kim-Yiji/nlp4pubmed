# oa_downloader.py
import os
import requests
import urllib.request
from xml.etree import ElementTree as ET


def fetch_oa_metadata(pmc_id: str) -> ET.Element:
    """
    PMC ID로부터 Open Access 논문 메타데이터(XML)를 가져옴
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"❌ 요청 실패: {response.status_code}")
    root = ET.fromstring(response.text)
    return root


def extract_download_link(root: ET.Element, file_format: str = "tgz") -> str | None:
    """
    메타데이터에서 원하는 포맷(tgz, pdf 등)의 다운로드 링크 추출
    """
    record = root.find(".//record")
    if record is None:
        return None

    for link in record.findall("link"):
        if link.attrib.get("format") == file_format:
            return link.attrib.get("href")

    return None


def download_file(href: str, save_dir: str = "Database/oa_api") -> str:
    """
    링크를 통해 파일 다운로드 및 저장
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = href.split("/")[-1]
    save_path = os.path.join(save_dir, filename)

    try:
        if href.startswith("ftp://"):
            with urllib.request.urlopen(href) as resp, open(save_path, "wb") as out_file:
                out_file.write(resp.read())
        else:
            file_data = requests.get(href)
            with open(save_path, "wb") as out_file:
                out_file.write(file_data.content)
    except Exception as e:
        raise RuntimeError(f"❌ 다운로드 실패: {e}")

    return save_path

def run_oa_api(pmc_id:str = None):

    if pmc_id is None:
        pmc_id = input("OA Web API로 저장할 PMC ID를 입력하세요 (예: PMC10203021): ").strip()
        if not pmc_id:
            print("❗ PMC ID를 입력해야 합니다.")
            return

    try:
        root = fetch_oa_metadata(pmc_id)
        href = extract_download_link(root, file_format="tgz")

        if href:
            saved_path = download_file(href)
            print(f"✅ 저장 완료: {saved_path}")
        else:
            print("❌ 원하는 포맷의 링크를 찾을 수 없습니다.")
    except Exception as e:
        print(e)
