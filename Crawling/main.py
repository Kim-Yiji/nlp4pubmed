from Crawling.eutilities import run_eutilities
from Crawling.bioc import run_bioc
from Crawling.oa_api import run_oa_api
from Crawling.ftp import run_ftp_download
from Crawling.aws import run_aws_sync

print("🔎 사용할 수집 방식 선택:")
print("1. E-utilities (Entrez)")
print("2. BioC API (PMC ID 입력)")
print("3. OA Web API")
print("4. FTP 다운로드")
print("5. AWS S3 동기화")

mode = input("번호 입력 (1~5): ").strip()
searchword = ""  # ✅ 여기에 고정 쿼리를 설정 (None이면 사용자 입력받음)s
pmc_id = ""
    # pmc_id = "PMC10203021"

if mode == "1":
    if searchword == "":
        run_eutilities()  # 검색어 직접 입력
    else:
        run_eutilities(searchword)
elif mode == "2":
    if searchword == "":
        pmc_id = input("BioC로 저장할 PMC ID를 입력하세요 (예: PMC7395540): ").strip()
    else:
        pmc_id = searchword
    run_bioc(pmc_id)
elif mode == "3":
    if pmc_id == "":
        run_oa_api()
    else:
        run_oa_api(pmc_id)
elif mode == "4":
    run_ftp_download()
elif mode == "5":
    run_aws_sync()
else:
    print("❗ 잘못된 입력입니다.")