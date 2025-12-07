import streamlit as st

# Başlık
st.title("Smart Travel Planner V1")

# Açıklama
st.write("This is a placeholder for V1 Streamlit UI.")

# Kullanıcıdan giriş al
user_input = st.text_area("Enter your travel preferences:")

# Girilen metni göster
st.write("You entered:", user_input)