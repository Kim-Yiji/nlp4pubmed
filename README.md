# nlp4pubmed ChatVTM mock up app
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## 앱 실행 명령어
앱을 실행하려면 터미널에 다음과 같이 입력해주세요
1. pip install streamlit
2. streamlit run streamlit_app.py
그럼 자동으로 브라우저 탭이 열립니다. 앱을 종료하려면 Ctrl + C를 눌러주세요. 

### 안내사항
1. 입력은 증상 > 나이 > 성별 > 신장 > 기타 정보 순으로 이루어집니다. 챗봇의 질문에 따라 입력하시면 됩니다. 
2. 증상에는 "피부 건조", "탈모", "근육 저림", 피로" 중에 하나를 입력해주세요. 기타 다른 값을 입력하셔도 되지만, 결과값이 "피부 건조"와 동일하게 나옵니다.
3. 나이에는 숫자만 입력해주세요- 예: 23
4. 성별은 여성이면 "여", 남성이면 "남"만 입력해주세요.
5. 신장은 숫자만 입력해주세요 - 예: 170
6. 기타 정보는 없으면 "없음", 있으면 "흡연 중" 등으로 입력해주세요. 
7. 추천 복용량이 나온 뒤 재시작하고 싶으면 증상부터 다시 입력하시면 됩니다. 

### 참고: 더미 데이터
더미 데이터는 다음과 같이 넣어두었습니다:
"피부 건조": {"vitamin": "vitamin C", "rda": "90mg", "url": "https://www.ncbi.nlm.nih.gov/books/NBK225480/#:~:text=To%20provide%20antioxidant%20protection%2C%20a,minimal%20urinary%20excretion%20of%20ascorbate."},
"탈모": {"vitamin": "Biotin", "rda": "30mcg", "url": "https://pubmed.ncbi.nlm.nih.gov/23193625/"},
"근육 저림": {"vitamin": "Calcium", "rda": "1000mg", "url": "https://pubmed.ncbi.nlm.nih.gov/21118827/"},
"피로": {"vitamin": "Vitamin B", "rda": "1.1mg", "url": "https://www.ncbi.nlm.nih.gov/books/NBK114296/"}