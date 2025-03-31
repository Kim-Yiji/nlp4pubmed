import os
import urllib.request

def run_ftp_download():
    """
    PMC FTP 서버에서 oa_bulk 파일 일부를 다운로드
    """
    base_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"
    test_file = "comm_use.A-B.xml.tar.gz"
    full_url = base_url + test_file

    save_dir = os.path.join("Database", "ftp")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, test_file)

    try:
        print(f"⬇️ 다운로드 중: {test_file}")
        urllib.request.urlretrieve(full_url, save_path)
        print(f"✅ 저장 완료: {save_path}")
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")


# ftp 형식 다운로드 해주는 함수
def download_via_ftp(url: str, save_path: str) -> None:
    """
    주어진 FTP URL에서 파일을 다운로드하여 지정한 경로에 저장
    """
    try:
        urllib.request.urlretrieve(url, save_path)
        # print(f"✅ 저장 완료 (FTP): {save_path}")
    except Exception as e:
        print(f"❌ FTP 다운로드 실패: {e}")