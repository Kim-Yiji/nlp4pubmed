import streamlit as st
import time

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.step = 0
    st.session_state.symptom = ""
    st.session_state.age = ""
    st.session_state.gender = ""
    st.session_state.height = ""
    st.session_state.other = ""
    st.session_state.typing = False
    # Add welcome message
    st.session_state.messages.append({"role": "assistant", "content": "안녕하세요, 어디가 불편하신가요? 증상을 입력해주세요."})

# Mock dataset
MOCK_DATASET = {
    "피부 건조": {"vitamin": "vitamin C", "rda": "90mg", "url": "https://www.ncbi.nlm.nih.gov/books/NBK225480/#:~:text=To%20provide%20antioxidant%20protection%2C%20a,minimal%20urinary%20excretion%20of%20ascorbate."},
    "탈모": {"vitamin": "Biotin", "rda": "30mcg", "url": "https://pubmed.ncbi.nlm.nih.gov/23193625/"},
    "근육 저림": {"vitamin": "Calcium", "rda": "1000mg", "url": "https://pubmed.ncbi.nlm.nih.gov/21118827/"},
    "피로": {"vitamin": "Vitamin B", "rda": "1.1mg", "url": "https://www.ncbi.nlm.nih.gov/books/NBK114296/"}
}

def typewriter_effect(text, container):
    """Display text with typewriter effect"""
    displayed_text = ""
    text_placeholder = container.empty()
    
    for char in text:
        displayed_text += char
        text_placeholder.markdown(displayed_text + "▋")  # Show cursor
        time.sleep(0.03)  # Adjust speed here (lower = faster)
    
    # Remove cursor and show final text
    text_placeholder.markdown(displayed_text)

def get_next_prompt(step):
    if step == 1:
        return "만 나이를 입력해주세요. 숫자만 입력해주세요."
    elif step == 2:
        return "성별을 입력해주세요"
    elif step == 3:
        return "신장(cm)을 입력해주세요."
    elif step == 4:
        return "이 외에 제가 고려해야 할 사항이 있을까요? 예: 흡연 여부, 임신 여부 등"
    return ""

def generate_final_response(symptom, age, gender, height, other):
    # Get matching vitamin data or default to vitamin C
    data = MOCK_DATASET.get(symptom, {"vitamin": "vitamin C", "rda": "90mg", "url": "https://www.ncbi.nlm.nih.gov/books/NBK225480/#:~:text=To%20provide%20antioxidant%20protection%2C%20a,minimal%20urinary%20excretion%20of%20ascorbate."})
    
    if other == "없음":
        return f"{symptom} 증상을 가진 당신에게 필요한 영양성분은 {data['vitamin']}입니다. {age}세 {height}cm {gender}성의 권장복용량(RDA)은 하루 {data['rda']}입니다. 더 자세한 사항은 다음 링크에서 확인 가능합니다: {data['url']}"
    else:
        return f"{symptom} 증상을 가진 당신에게 필요한 영양성분은 {data['vitamin']}입니다. {other} 여부를 고려했을 때, {age}세 {height}cm {gender}성의 권장복용량(RDA)은 하루 {data['rda']}입니다. 더 자세한 사항은 다음 링크에서 확인 가능합니다: {data['url']}"

# Streamlit UI
st.title("🩺 ChatVTM")

# Basic info section
st.markdown("당신의 개인 약사 ChatVTM입니다. 증상과 개인 정보를 입력하면 당신에게 필요한 영양성분 및 권장복용량(RDA)을 추천드립니다.")

# Grey box for second paragraph
st.markdown("""
<div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin: 10px 0;">
💊 RDA는 Recommended Dietary Allowances의 약자로, 미국 국립보건원 NIH 산하의 식품영양위원회(Food and Nutrition Board)가 설정한 필수 영양소 섭취 기준입니다. 대부분의 건강한 사람들의 영양 요구를 충족하기에 충분한 수준으로 판단되는 섭취량을 나타내며, 성별, 연령, 신장에 따라 복용량이 달라집니다.
</div>
""", unsafe_allow_html=True)

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1 and st.session_state.typing:
            # Apply typewriter effect to the latest assistant message
            typewriter_effect(message["content"], st)
            st.session_state.typing = False
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("답변을 입력하세요..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process user input based on current step
    if st.session_state.step == 0:
        # Store symptom
        st.session_state.symptom = prompt
        st.session_state.step = 1
        next_prompt = get_next_prompt(1)
        st.session_state.messages.append({"role": "assistant", "content": next_prompt})
        st.session_state.typing = True
        
    elif st.session_state.step == 1:
        # Store age
        st.session_state.age = prompt
        st.session_state.step = 2
        next_prompt = get_next_prompt(2)
        st.session_state.messages.append({"role": "assistant", "content": next_prompt})
        st.session_state.typing = True
        
    elif st.session_state.step == 2:
        # Store gender
        st.session_state.gender = prompt
        st.session_state.step = 3
        next_prompt = get_next_prompt(3)
        st.session_state.messages.append({"role": "assistant", "content": next_prompt})
        st.session_state.typing = True
        
    elif st.session_state.step == 3:
        # Store height and ask for other considerations
        st.session_state.height = prompt
        st.session_state.step = 4
        next_prompt = get_next_prompt(4)
        st.session_state.messages.append({"role": "assistant", "content": next_prompt})
        st.session_state.typing = True
        
    elif st.session_state.step == 4:
        # Store other considerations and generate final response
        st.session_state.other = prompt
        final_response = generate_final_response(
            st.session_state.symptom, 
            st.session_state.age, 
            st.session_state.gender, 
            st.session_state.height,
            st.session_state.other
        )
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.session_state.typing = True
        # Add restart prompt
        restart_prompt = "다른 증상이 있으신가요? 재시작하려면 다른 증상을 입력해주세요."
        st.session_state.messages.append({"role": "assistant", "content": restart_prompt})
        st.session_state.step = 5  # Ready for restart
        
    elif st.session_state.step == 5:
        # Restart session with new symptom
        st.session_state.symptom = prompt
        st.session_state.age = ""
        st.session_state.gender = ""
        st.session_state.height = ""
        st.session_state.other = ""
        st.session_state.step = 1
        next_prompt = get_next_prompt(1)
        st.session_state.messages.append({"role": "assistant", "content": next_prompt})
        st.session_state.typing = True
    
    # Rerun to update the display
    st.rerun()

# Optional: Display collected data for debugging (you can remove this)
if st.session_state.step > 0:
    with st.sidebar:
        st.write("수집된 정보:")
        if st.session_state.symptom:
            st.write(f"증상: {st.session_state.symptom}")
        if st.session_state.age:
            st.write(f"나이: {st.session_state.age}")
        if st.session_state.gender:
            st.write(f"성별: {st.session_state.gender}")
        if st.session_state.height:
            st.write(f"신장: {st.session_state.height}")
        if st.session_state.other:
            st.write(f"기타 고려사항: {st.session_state.other}")