import streamlit as st
import torch
import cv2
import numpy as np
from detect_and_localize import detect_multiple_stars, load_models
from PIL import Image
import tempfile
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.title("⭐ Star Detection & Brightness Estimation")

# === Sidebar ayarlar ===
prob_thr = st.sidebar.slider("Classifier Threshold", 0.0, 1.0, 0.2, 0.01)
stride = st.sidebar.slider("Stride (px)", 4, 32, 8, 4)

uploaded_file = st.file_uploader("Bir yıldız görüntüsü yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Geçici dosyaya kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.getvalue())
        image_path = temp.name

    # Görseli göster
    st.image(uploaded_file, caption="Yüklenen Görsel", use_column_width=True)

    # Cihaz ve modelleri yükle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, regressor = load_models(device)

    # Yıldızları tespit et
    stars = detect_multiple_stars(image_path, device, classifier, regressor,
                                   patch_size=32, stride=stride, prob_thr=prob_thr)

    if not stars:
        st.warning("❌ Yıldız bulunamadı.")
    else:
        # Orijinal görseli yükle ve çizim için kopyasını al
        image = cv2.imread(image_path)
        annotated_image = image.copy()
                # 🎯 Parlaklık histogramı
        brightness_values = [star["brightness"] for star in stars]

        fig, ax = plt.subplots()
        ax.hist(brightness_values, bins=20, color='skyblue', edgecolor='black')
        ax.set_title("Parlaklık Dağılımı (Brightness Histogram)")
        ax.set_xlabel("Parlaklık")
        ax.set_ylabel("Yıldız Sayısı")
        st.pyplot(fig)


        for i, star in enumerate(stars):
            x, y = star["pos"]
            center = (int(round(x)), int(round(y)))
            cv2.circle(annotated_image, center, radius=5, color=(0, 255, 255), thickness=1)
            cv2.putText(annotated_image, f"Y{i+1}", (center[0]+5, center[1]-5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 255), thickness=1)

        # RGB'ye çevir ve göster
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image, caption="Tespit Edilen Yıldızlar", use_column_width=True)

        # Bilgileri yazdır
        for i, star in enumerate(stars):
            st.markdown(f"""
            ---
            ### ⭐ Yıldız {i+1}
            📍 **Pozisyon:** {star['pos']}  
            📐 **Ofset:** {star['offset']}  
            🌟 **Parlaklık:** {star['brightness']:.3f}  
            """)
