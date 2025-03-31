from Bio import Entrez

def get_search_query_and_count(email: str, query: str | None = None) -> tuple[str, int]:
    """
    PubMed 검색어를 받아 결과 개수를 반환한다.
    query가 None이면 사용자에게 입력을 요청한다.

    Args:
        email (str): Entrez API에 사용할 이메일
        query (str | None): 검색어. None이면 입력 받음.

    Returns:
        tuple: (검색어, 검색 결과 개수)
    """
    Entrez.email = email

    # query가 None이면 사용자 입력 받기
    if query is None:
        query = input("🔍 PubMed에서 검색할 키워드를 입력하세요: ").strip()
        if not query:
            print("❗ 검색어를 입력해야 합니다.")
            exit()

    # 검색 요청
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    total_count = int(search_results["Count"])
    return query, total_count