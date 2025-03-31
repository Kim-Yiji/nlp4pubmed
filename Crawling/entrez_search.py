from Bio import Entrez

def get_search_query_and_count(email: str) -> tuple[str, int]:
    """
    사용자로부터 검색어를 입력받고, PubMed에서 해당 검색어의 검색 결과 개수를 반환

    Args:
        email (str): Entrez API 사용을 위한 이메일

    Returns:
        tuple: (검색어, 검색 결과 개수)
    """
    Entrez.email = email

    query = input("🔍 PubMed에서 검색할 키워드를 입력하세요: ").strip()
    if not query:
        print("❗ 검색어를 입력해야 합니다.")
        exit()

    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    total_count = int(search_results["Count"])
    return query, total_count