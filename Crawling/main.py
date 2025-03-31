from Crawling.eutilities import run_eutilities
from Crawling.bioc import run_bioc

print("🔎 사용할 수집 방식 선택:")
print("1. E-utilities (Entrez)")
print("2. BioC API")

mode = input("번호 입력 (1 or 2): ").strip()

if mode == "1":
    run_eutilities()  # 검색어 직접 입력
elif mode == "2":
    pmc_id = input("BioC로 저장할 PMC ID를 입력하세요 (예: PMC7395540): ").strip()
    run_bioc(pmc_id)
else:
    print("❗ 잘못된 입력입니다.")