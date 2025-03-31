import os
import requests
from xml.etree import ElementTree as ET


def fetch_oai_metadata(pmc_id: str) -> ET.Element:
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


def save_download_link_to_txt(pmc_id: str, href: str, save_dir: str = "Database/oai_api") -> None:
    """
    다운로드 링크를 txt 파일에 저장
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{pmc_id}_download_link.txt"
    save_path = os.path.join(save_dir, filename)

    try:
        with open(save_path, "w") as file:
            file.write(f"PMC ID: {pmc_id}\n")
            file.write(f"Download Link: {href}\n")
        print(f"✅ 다운로드 링크가 {filename}에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 링크 저장 실패: {e}")


def run_oai_api(pmc_id: str = None):
    """
    OAI API를 사용하여 PMC ID에 대한 다운로드 링크를 가져오고 txt 파일로 저장
    """
    if pmc_id == None:
        pmc_id = input("OAI API로 검색할 PMC ID를 입력하세요 (예: PMC7395540): ").strip()

    try:
        root = fetch_oai_metadata(pmc_id)
        href = extract_download_link(root, file_format="tgz")

        if href:
            save_download_link_to_txt(pmc_id, href)
        else:
            print("❌ 원하는 포맷의 링크를 찾을 수 없습니다.")
    except Exception as e:
        print(e)


# 사용 예시
# run_oai_api("PMC1234567")  # 특정 PMC ID 입력