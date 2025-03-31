import os
import subprocess

def run_aws_sync():
    """
    AWS S3에서 PMC Open Access Subset 일부 동기화
    (aws-cli 필요, 인증 없이 가능)
    """
    local_path = os.path.join("Database", "aws")
    os.makedirs(local_path, exist_ok=True)

    # 테스트용: 작은 부분만 가져오기
    s3_path = "s3://pmc-oa-opendata/oa_comm/xml/all/"
    command = [
        "aws", "s3", "sync", s3_path, local_path,
        "--no-sign-request",
        "--exclude", "*", "--include", "PMC7*.nxml"
    ]

    print("☁️ AWS S3에서 일부 파일을 동기화합니다 (PMC7xxx)")
    try:
        subprocess.run(command, check=True)
        print("✅ 동기화 완료!")
    except subprocess.CalledProcessError as e:
        print(f"❌ AWS 동기화 실패: {e}")