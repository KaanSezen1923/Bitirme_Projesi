import streamlit as st
import requests
import json

# Sayfa Ayarları
st.set_page_config(
    page_title="Süt Sihirbazı - Test Paneli",
    page_icon="🥛",
    layout="wide"
)

# Başlık ve Açıklama
st.title("🥛 Süt Sihirbazı Prototip")
st.markdown("""
Bu panel, **FastAPI backend** ile iletişim kurarak RAG ve Text-to-SQL performansını test etmek için tasarlanmıştır.
""")

# Yan Menü (Sidebar) - API Ayarları
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
    
    if st.button("API Bağlantısını Test Et"):
        try:
            response = requests.get(f"{api_url}/")
            if response.status_code == 200:
                st.success(f"Bağlantı Başarılı: {response.json().get('message')}")
            else:
                st.error("API'ye ulaşıldı ancak hata döndü.")
        except requests.exceptions.ConnectionError:
            st.error("API'ye bağlanılamadı. Backend'in çalıştığından emin olun.")

# Chat Geçmişini Başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş Mesajları Ekrana Bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Eğer geçmiş mesajda debug verisi varsa onu da göster (Expanders kapalı gelir)
        if "debug_info" in message:
            with st.expander("🛠️ Teknik Detaylar (SQL & Sınıflandırma)"):
                st.json(message["debug_info"])

# Kullanıcı Girdisi
if prompt := st.chat_input("Sorunuzu buraya yazın (Örn: Çiftlikte kaç inek var?)..."):
    
    # 1. Kullanıcı mesajını ekrana bas ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. API'ye İstek At
    with st.chat_message("assistant"):
        with st.spinner("Süt Sihirbazı düşünüyor..."):
            try:
                # FastAPI endpoint'ine POST isteği
                payload = {"question": prompt}
                response = requests.post(f"{api_url}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    answer_text = data.get("answer", "Cevap alınamadı.")
                    
                    # Cevabı yazdır
                    st.markdown(answer_text)
                    
                    # Teknik Detayları Hazırla
                    debug_info = {
                        "Sınıflandırma": data.get("classification"),
                        "SQL Sorgusu": data.get("sql_query"),
                        "SQL Sonucu": data.get("sql_result") # Ham veri
                    }
                    
                    # Expander içinde teknik detayları göster (Test için kritik)
                    with st.expander("🛠️ Teknik Detaylar (Debug)"):
                        st.write(f"**Mod:** `{debug_info['Sınıflandırma']}`")
                        
                        if debug_info["SQL Sorgusu"]:
                            st.caption("Üretilen SQL:")
                            st.code(debug_info["SQL Sorgusu"], language="sql")
                        
                        if debug_info["SQL Sonucu"]:
                            st.caption("Veritabanından Dönen Ham Veri:")
                            st.code(debug_info["SQL Sonucu"])

                    # Asistan cevabını ve debug verisini geçmişe kaydet
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer_text,
                        "debug_info": debug_info
                    })
                    
                else:
                    st.error(f"API Hatası: {response.status_code}")
                    st.text(response.text)
            
            except requests.exceptions.ConnectionError:
                st.error("API'ye bağlanılamadı. Lütfen 'uvicorn api:app' komutu ile backend'i çalıştırdığınızdan emin olun.")
            except Exception as e:
                st.error(f"Beklenmeyen bir hata oluştu: {e}")