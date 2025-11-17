# app.py
import streamlit as st

st.set_page_config(page_title="Brotherhood Hub", layout="wide")

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih menu:", ["Dashboard", "Settings", "Logs"])

# Main area
st.title("Brotherhood Hub")
st.write(f"Menu dipilih: {menu}")

# Example button
if st.button("Contoh Button"):
    st.write("Button ditekan!")

# Example file uploader
uploaded_file = st.file_uploader("Upload fail untuk analisis nanti", type=["png","jpg","csv"])
if uploaded_file:
    st.write(f"Fail diterima: {uploaded_file.name}")
