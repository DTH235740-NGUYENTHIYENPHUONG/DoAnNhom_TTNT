import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os

# Cấu hình trang web
st.set_page_config(
    page_title="Hệ Thống Chẩn Đoán Bệnh Lá Xoài",
    page_icon="🥭",
    layout="wide"
)

# ================= 1. KHAI BÁO DỮ LIỆU CHUYÊN MÔN =================
CLASS_NAMES = ['Anthracnose', 'Die_Back', 'Healthy', 'Powdery_Mildew']

DISEASE_INFO = {
    'Anthracnose': {
        'name_vn': 'Bệnh Thán Thư (Anthracnose)',
        'status': '⚠️ CẢNH BÁO BỆNH',
        'color': '#E74C3C',
        'cause': 'Do nấm *Colletotrichum gloeosporioides* gây ra, phát triển mạnh trong điều kiện thời tiết nóng ẩm, mưa nhiều hoặc sương mù dày.',
        'symptoms': 'Xuất hiện các đốm màu nâu đen có hình dạng bất định trên lá. Khi bệnh nặng, các vết bệnh liên kết lại làm cháy khô từng mảng lớn, lá dễ bị biến dạng và rụng sớm.',
        'treatment': [
            '**Canh tác:** Cắt tỉa cành tạo độ thông thoáng cho tán cây; thu gom và tiêu hủy triệt để lá, cành bị bệnh rụng dưới gốc.',
            '**Hóa học:** Phun phòng trừ vào giai đoạn cây ra đọt non và ra hoa bằng các hoạt chất như *Azoxystrobin, Difenoconazole, Mancozeb* hoặc *Propineb* (luân phiên thuốc để tránh kháng thuốc).'
        ]
    },
    'Die_Back': {
        'name_vn': 'Bệnh Chết Ngọn / Khô Cành (Die Back)',
        'status': '🚨 NGUY HIỂM CAO',
        'color': '#8E44AD',
        'cause': 'Do nấm *Lasiodiplodia theobromae* xâm nhập qua các vết thương cơ giới, vết côn trùng chích hút hoặc sau khi cây bị suy yếu.',
        'symptoms': 'Lá bắt đầu khô cháy từ chóp lá và mép lá lan dần vào trong. Cành non chuyển sang màu nâu sẫm, héo rũ và chết khô từ ngọn lan xuống thân chính.',
        'treatment': [
            '**Canh tác:** Cắt bỏ ngay các cành bị khô (cắt sâu xuống phần mô gỗ khỏe mạnh ít nhất 10-15cm), sau đó quét thuốc gốc Đồng hoặc vôi vào vết cắt.',
            '**Hóa học & Dinh dưỡng:** Bón phân cân đối, tăng cường Kali và phân hữu cơ vi sinh để phục hồi sức đề kháng; phun các thuốc chứa *Carbendazim, Thiophanate-methyl* hoặc *Copper Oxychloride*.'
        ]
    },
    'Healthy': {
        'name_vn': 'Lá Khỏe Mạnh (Healthy)',
        'status': '✅ AN TOÀN',
        'color': '#27AE60',
        'cause': 'Cây sinh trưởng tốt, không ghi nhận dấu hiệu xâm nhiễm của nấm bệnh hay vi khuẩn.',
        'symptoms': 'Phiến lá xanh bóng đồng đều, gân lá rõ nét, bề mặt không có đốm hoại tử, lớp phấn trắng hay hiện tượng cháy mép lá.',
        'treatment': [
            '**Duy trì chăm sóc:** Tiếp tục chế độ tưới nước hợp lý, bón phân định kỳ theo từng giai đoạn sinh trưởng của cây xoài.',
            '**Phòng ngừa:** Thăm vườn thường xuyên vào mùa mưa để phát hiện sớm các dấu hiệu bất thường trên đọt non.'
        ]
    },
    'Powdery_Mildew': {
        'name_vn': 'Bệnh Phấn Trắng (Powdery Mildew)',
        'status': '⚠️ CẢNH BÁO BỆNH',
        'color': '#D35400',
        'cause': 'Do nấm *Oidium mangiferae* gây ra, thường bùng phát mạnh khi thời tiết mát mẻ, ẩm độ không khí cao nhưng ít mưa lớn.',
        'symptoms': 'Bề mặt lá non, cuống lá hoặc phát hoa phủ một lớp bột phấn màu trắng xám như rắc bột. Lá bị bệnh thường xoăn lại, biến dạng, khô và rụng hàng loạt.',
        'treatment': [
            '**Canh tác:** Tỉa bớt cành vượt, cành rậm rạp trong tán để ánh nắng chiếu vào; tránh bón thừa phân đạm (N) khi cây đang ra lá non.',
            '**Hóa học:** Phun các loại thuốc đặc trị nấm phấn trắng chứa hoạt chất *Hexaconazole, Myclobutanil, Sulfur (Lưu huỳnh)* hoặc *Triadimefon* ngay khi thấy vết phấn đầu tiên.'
        ]
    }
}

# ================= 2. HÀM TẢI MÔ HÌNH (SỬA LỖI TƯƠNG THÍCH KERAS 3) =================
@st.cache_resource
def load_mango_model():
    model_path = "mobilenetv2_mango_final.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file `{model_path}` trong thư mục hiện tại (`{os.getcwd()}`). Vui lòng kiểm tra lại tên file!")
        st.stop()

    # Vá lỗi quantization_config cho các phiên bản Keras khác nhau
    original_from_config = tf.keras.layers.Dense.from_config
    @classmethod
    def patched_from_config(cls, config):
        config.pop('quantization_config', None)
        return original_from_config(config)
    tf.keras.layers.Dense.from_config = patched_from_config

    # Định nghĩa lại 2 lớp tiền xử lý bên trong MobileNetV2
    class TrueDivideLayer(tf.keras.layers.Layer):
        def call(self, inputs, y=127.5):
            return tf.math.truediv(inputs, tf.cast(y, inputs.dtype))

    class SubtractLayer(tf.keras.layers.Layer):
        def call(self, inputs, y=1.0):
            return tf.math.subtract(inputs, tf.cast(y, inputs.dtype))

    custom_objects = {
        'TrueDivide': TrueDivideLayer,
        'Subtract': SubtractLayer
    }

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"❌ Lỗi chi tiết khi nạp mô hình: {e}")
        st.stop()

# ================= 3. HÀM XỬ LÝ TÁCH NỀN TÙY CHỌN =================
def remove_background_to_white(pil_image):
    from rembg import remove
    rgba_img = remove(pil_image)
    white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, rgba_img)
    return composite.convert("RGB")

# ================= 4. GIAO DIỆN NGƯỜI DÙNG =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2909/2909761.png", width=80)
    st.title("Cấu Hình Hệ Thống")
    st.markdown("---")
    
    # Đặt mặc định False để giữ nguyên ảnh gốc giống lúc train
    use_bg_removal = st.checkbox(
        "Tự động tách nền trắng (rembg)", 
        value=False, 
        help="Chỉ bật nếu ảnh chụp có nền quá rối và lá còn nguyên vẹn. Tắt nếu lá bị cháy khô ở viền."
    )
    
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy tối thiểu (%)", 
        min_value=30, max_value=95, value=60, step=5
    )
    
    st.markdown("---")
    st.info(
        "**Thông tin mô hình:**\n"
        "- Kiến trúc: MobileNetV2\n"
        "- Kích thước đầu vào: 224x224 px\n"
        "- Số lớp nhận diện: 4 nhóm"
    )

st.title("🥭 HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH TRÊN LÁ XOÀI")
st.markdown("Tải lên hình ảnh lá xoài để hệ thống AI phân tích triệu chứng, xác định loại bệnh và đề xuất phác đồ xử lý.")
st.markdown("---")

uploaded_file = st.file_uploader("Chọn ảnh lá xoài từ thiết bị (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_mango_model()
    
    # Sửa lỗi xoay ngang ảnh khi chụp từ điện thoại
    raw_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    
    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.subheader("🖼️ Hình ảnh đầu vào")
        if use_bg_removal:
            with st.spinner("Đang tách nền ảnh..."):
                try:
                    processed_img = remove_background_to_white(raw_image)
                    st.image(processed_img, caption="Ảnh đã tách nền trắng", use_container_width=True)
                except Exception as e:
                    st.warning(f"Không thể tách nền ({e}), sử dụng ảnh gốc.")
                    processed_img = raw_image
                    st.image(processed_img, caption="Ảnh gốc", use_container_width=True)
        else:
            processed_img = raw_image
            st.image(processed_img, caption="Ảnh lá xoài tải lên", use_container_width=True)

    with col2:
        st.subheader("🔬 Kết quả phân tích AI")
        
        with st.spinner("Đang chạy mô hình chẩn đoán..."):
            img_resized = processed_img.resize((224, 224))
            # Giữ nguyên dải [0, 255] vì model đã tích hợp sẵn lớp chuẩn hóa [-1, 1]
            img_array = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_batch, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(predictions[predicted_idx]) * 100
            
            info = DISEASE_INFO[predicted_class]

        # Hiển thị kết quả chính
        st.markdown(
            f"""
            <div style="padding: 18px; border-radius: 10px; border-left: 6px solid {info['color']}; background-color: rgba(128,128,128,0.1);">
                <span style="font-size: 14px; font-weight: bold; color: {info['color']};">{info['status']}</span>
                <h2 style="margin: 5px 0 10px 0;">{info['name_vn']}</h2>
                <p style="margin: 0; font-size: 16px;">Độ tin cậy dự đoán: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_