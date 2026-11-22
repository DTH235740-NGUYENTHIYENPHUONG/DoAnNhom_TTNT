import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Kiểm tra thư viện tách nền rembg (tránh lỗi nếu chưa cài)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# 1. Cấu hình trang Web
st.set_page_config(page_title="Chẩn đoán Bệnh Lá Xoài - MobileNetV2", page_icon="🥭", layout="wide")

# 2. Danh mục 4 lớp bệnh và khuyến cáo xử lý
CLASS_NAMES = ['Anthracnose', 'Die_Back', 'Healthy', 'Powdery_Mildew']

CLASS_INFO = {
    'Anthracnose': {
        'name_vi': 'Bệnh Thán Thư',
        'desc': 'Các đốm màu nâu đen lan rộng trên lá, gây khô xơ và rụng lá.',
        'treatment': 'Phun thuốc gốc Đồng (Copper), Mancozeb hoặc Antracol 70WP.'
    },
    'Die_Back': {
        'name_vi': 'Bệnh Chết Đọt / Cháy Lá',
        'desc': 'Cành non bị khô đen từ ngọn xuống, lá héo rụi và cháy xém.',
        'treatment': 'Cắt bỏ cành bệnh, phun thuốc chứa Carbendazim hoặc gốc Đồng.'
    },
    'Healthy': {
        'name_vi': 'Lá Khỏe Mạnh',
        'desc': 'Bề mặt lá xanh tươi, bóng mượt, không có dấu hiệu vết bệnh.',
        'treatment': 'Tiếp tục duy trì chế độ tưới nước và bón phân định kỳ.'
    },
    'Powdery_Mildew': {
        'name_vi': 'Bệnh Phấn Trắng',
        'desc': 'Lớp phấn mỏng màu trắng xám bao phủ trên bề mặt lá non.',
        'treatment': 'Phun thuốc chứa Lưu huỳnh (Sulfur), Anvil 5SC hoặc Hexaconazole.'
    }
}

# Tạo 2 lớp tương thích để Keras 3 đọc được phép chia và trừ trong mô hình
class TrueDivideLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, inputs, y=127.5, *args, **kwargs):
        self.y = y
        return super().__call__(inputs, *args, **kwargs)
    def call(self, inputs):
        return tf.math.truediv(inputs, self.y)

class SubtractLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, inputs, y=1.0, *args, **kwargs):
        self.y = y
        return super().__call__(inputs, *args, **kwargs)
    def call(self, inputs):
        return tf.math.subtract(inputs, self.y)

# 3. Nạp mô hình MobileNetV2
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenetv2_mango_final.keras",
        custom_objects={'TrueDivide': TrueDivideLayer, 'Subtract': SubtractLayer},
        compile=False
    )

model = load_model()

# Hàm tách nền phức tạp và đưa lá xoài về nền trắng chuẩn phòng thí nghiệm
def process_white_background(pil_image):
    rgba_img = pil_image.convert("RGBA")
    no_bg = remove(rgba_img)
    white_bg = Image.new("RGBA", no_bg.size, "WHITE")
    clean_img = Image.alpha_composite(white_bg, no_bg).convert("RGB")
    return clean_img

# 4. Thanh bên (Sidebar) cấu hình tiền xử lý
with st.sidebar:
    st.header("⚙️ Cấu hình Xử lý Ảnh")
    use_bg_removal = st.toggle(
        "Tự động tách nền trắng",
        value=True,
        help="Bật khi kiểm tra ảnh chụp ngoài vườn có nền cây cối/đất đá phức tạp."
    )
    if use_bg_removal and not REMBG_AVAILABLE:
        st.warning("⚠️ Chưa cài thư viện `rembg`. Hãy chạy: `pip install rembg onnxruntime`")

    st.markdown("---")
    st.subheader("📌 Hướng dẫn chụp ảnh chuẩn")
    st.markdown("""
    - Chụp cận cảnh **1 chiếc lá** rõ nét.
    - Tránh bóng râm quá tối hoặc ánh nắng chói lóa trên mặt lá.
    - Bật **Tự động tách nền** nếu ảnh chụp trực tiếp ngoài tự nhiên.
    """)

# 5. Giao diện chính
st.title("🥭 Hệ Thống Chẩn Đoán Bệnh Trên Lá Xoài")
st.caption("Ứng dụng mô hình Deep Learning MobileNetV2 (Transfer Learning) kết hợp Tiền xử lý Tách nền")
st.markdown("---")

uploaded_file = st.file_uploader("📸 Tải ảnh lá xoài cần kiểm tra (JPG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    # Xử lý tách nền nếu người dùng bật tùy chọn
    if use_bg_removal and REMBG_AVAILABLE:
        with st.spinner("Đang tách nền hậu cảnh ngoài vườn..."):
            processed_image = process_white_background(raw_image)
    else:
        processed_image = raw_image

    col1, col2, col3 = st.columns([1, 1, 1.2])
    
    with col1:
        st.image(raw_image, caption="1. Ảnh gốc tải lên", use_container_width=True)
        
    with col2:
        caption_text = "2. Ảnh đã tách nền trắng (Đưa vào AI)" if (use_bg_removal and REMBG_AVAILABLE) else "2. Ảnh đầu vào trực tiếp"
        st.image(processed_image, caption=caption_text, use_container_width=True)
        
    with col3:
        st.subheader("🔍 Kết quả Phân tích")
        
        # Tiền xử lý đúng chuẩn MobileNetV2 (sử dụng processed_image)
        img_resized = processed_image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Suy luận dự đoán
        preds = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx]) * 100
        
        info = CLASS_INFO[top_class]
        if top_class == 'Healthy':
            st.success(f"**Chẩn đoán**: {info['name_vi']}")
        else:
            st.error(f"**Chẩn đoán**: {info['name_vi']}")
            
        st.metric(label="Độ tin cậy AI", value=f"{confidence:.2f}%")
        
        if confidence < 65.0:
            st.warning("⚠️ Độ tin cậy chưa cao. Vui lòng thử chụp cận cảnh 1 lá rõ nét hơn.")
        
    # 6. Chi tiết xác suất 4 lớp & Khuyến cáo
    st.markdown("---")
    st.subheader("📊 Tỷ lệ xác suất 4 nhãn phân loại:")
    for idx, c_name in enumerate(CLASS_NAMES):
        prob = float(preds[idx])
        st.write(f"**{CLASS_INFO[c_name]['name_vi']}**: {prob*100:.2f}%")
        st.progress(prob)
        
    st.info(f"💡 **Mô tả triệu chứng**: {info['desc']}\n\n🛠️ **Khuyến cáo xử lý**: {info['treatment']}")