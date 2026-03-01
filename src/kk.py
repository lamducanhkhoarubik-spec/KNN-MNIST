import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN (DARK MODE & GLOW)
# ==========================================
st.set_page_config(page_title="DIGIT AI RECOGNITION", layout="wide")

TEN_CUA_ONG = "LÂM ĐỨC ANH KHOA" 

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&display=swap');

    /* Hiệu ứng load trượt từ dưới lên */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(50px); filter: blur(10px); }}
        to {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
    }}

    .stApp {{
        background-color: #020408;
        background-image: radial-gradient(circle at 2px 2px, rgba(0, 251, 255, 0.05) 1px, transparent 0);
        background-size: 40px 40px;
        animation: fadeInUp 1s ease-out;
    }}

    /* Tên ông ở góc phải - Sáng rực đỏ */
    .user-badge {{
        position: fixed;
        top: 20px; right: 20px;
        padding: 10px 20px;
        border: 2px solid #FF4B4B;
        color: #FF4B4B !important;
        text-shadow: 0 0 10px #FF4B4B;
        font-family: 'Orbitron', sans-serif;
        background: rgba(0,0,0,0.8);
        z-index: 9999;
        border-radius: 5px;
    }}

    /* Chữ nào cũng phải sáng */
    h1, h2, h3, label, p, .stMarkdown {{
        color: #00FBFF !important;
        text-shadow: 0 0 10px #00FBFF, 0 0 20px #00FBFF;
        font-family: 'Orbitron', sans-serif !important;
    }}

    .main-title {{
        text-align: center;
        font-size: 3.5rem !important;
        letter-spacing: 10px;
        margin-bottom: 40px;
    }}

    .result-card {{
        border: 2px solid #00FBFF;
        background: rgba(0, 251, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 251, 255, 0.2);
    }}
    </style>
    <div class="user-badge">BOSS: {TEN_CUA_ONG}</div>
    <h1 class="main-title">DIGIT RECOGNIZATION</h1>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL (GIỮ NGUYÊN RUỘT CŨ)
# ==========================================
@st.cache_resource
def init_model():
    (x_train, y_train), _ = mnist.load_data()
    x_train_final = x_train.reshape(-1, 784) / 255.0
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1) 
    knn.fit(x_train_final[:30000], y_train[:30000])
    return knn

knn = init_model()

def predict_logic(img):
    # Xử lý ảnh: Chuyển xám -> Nhị phân
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 2. KHÔNG DÙNG THRESH_BINARY_INV vì bảng vẽ đã là nền đen rồi!
    # Chỉ dùng THRESH_BINARY để làm sạch nét vẽ
    _, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)

    coords = cv.findNonZero(thresh)
    if coords is None: return None, "Không thấy nét vẽ"
    
    # 3. Cắt sát vùng có nét vẽ
    x, y, w, h = cv.boundingRect(coords)
    digit_crop = thresh[y:y+h, x:x+w]

    # 4. KHÔNG DÙNG DILATE (Làm dày nét) nếu nét vẽ đã đủ to
    # Dilation quá đà biến số 1 thành một cục tròn -> AI đoán là số 0
    digit_final = digit_crop 

    # 5. Resize về chuẩn 20x20 (giữ tỉ lệ)
    ratio = 20.0 / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    resized = cv.resize(digit_final, new_size, interpolation=cv.INTER_AREA)

    # 6. Căn giữa vào khung 28x28
    mask = np.zeros((28, 28), dtype=np.uint8)
    sx = (28 - new_size[0]) // 2
    sy = (28 - new_size[1]) // 2
    mask[sy:sy+new_size[1], sx:sx+new_size[0]] = resized

    # 7. Dự đoán
    final_input = mask.reshape(1, 784) / 255.0
    result = knn.predict(final_input)
    
    return mask, result[0]

# ==========================================
# 3. GIAO DIỆN WEB (BẢNG VẼ & UPLOAD)
# ==========================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("✍️ HỆ THỐNG VẼ TAY")
    # Bảng vẽ
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    st.markdown("---")
    st.subheader("📁 HOẶC UPLOAD FILE")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

with col_right:
    st.subheader("🧠 PHÂN TÍCH AI")
    
    # Ưu tiên xử lý từ bảng vẽ nếu có dữ liệu
    if canvas_result.image_data is not None:
        # Chuyển RGBA sang BGR (Bỏ kênh Alpha để tránh ra toàn số 0)
        img_canvas = canvas_result.image_data.astype(np.uint8)
        img_canvas_bgr = cv.cvtColor(img_canvas, cv.COLOR_RGBA2BGR)
        
        # Kiểm tra xem có nét vẽ thực sự không (tránh auto-predict nền đen)
        if np.any(img_canvas_bgr > 0):
            if st.button("CHẨN ĐOÁN BẢNG VẼ", use_container_width=True):
                processed_mask, label = predict_logic(img_canvas_bgr)
                if processed_mask is not None:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.image(processed_mask, width=150, caption="AI Vision (28x28)")
                    st.markdown(f"<h1 style='font-size:100px; margin:0;'>{label}</h1>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.balloons()

    # Xử lý file upload
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        st.image(image, caption="Ảnh ông giáo vừa quăng lên", width=200)
        
        if st.button("CHẨN ĐOÁN FILE ẢNH", use_container_width=True):
            processed_mask, label = predict_logic(img_cv)
            if processed_mask is not None:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(processed_mask, width=150)
                st.markdown(f"<h1 style='font-size:100px; margin:0;'>{label}</h1>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.3; margin-top:50px;'>CORE: KNN-MNIST | SYSTEM STATUS: ONLINE</p>", unsafe_allow_html=True)