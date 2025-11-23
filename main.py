import streamlit as st
import time
#显示
def mock_login(uname,pwd):
    time.sleep(3)
    return uname == "Jack" and pwd == '123'
username = st.text_input('Username','Jack')
password = st.text_input('Password','1234')

if st.button('Login'):
    with st.spinner('Loading..'):
        Login_result = mock_login(username,password)
        text = '登录成功' if Login_result else '登录失败'
        st.write(text)


















