import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Vô hiệu hóa GPU

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
import unicodedata

# --- Load model và các đối tượng cần thiết ---
model_path = 'model/ann_model.keras'
preprocessor = joblib.load("model/preprocessor_no_standarscaler.pkl")
feature_list = joblib.load("model/final_feature_list.pkl")

model = load_model(model_path, compile=False)
# --- Khởi tạo Flask ---
app = Flask(__name__)
legal_map = {
    'Sổ hồng': 'Have certificate',
    'Không có giấy tờ': 'None',
    'Hợp đồng mua bán': 'Sale contract'
}
furniture_map = {
    'Đầy đủ': 'Full',
    'Cơ bản': 'Basic',
    'Không có': 'None'
}
# Các danh sách hỗ trợ
province_mapping = {
    'TPHCM': 'Hồ Chí Minh', 'TpHCM': 'Hồ Chí Minh', 'TP. HCM': 'Hồ Chí Minh',
    'TP Hồ Chí Minh': 'Hồ Chí Minh', 'Hồ Chí Mính': 'Hồ Chí Minh', 'Hồ Chí Minh.': 'Hồ Chí Minh',
    'Hà Nội': 'Hà Nội', 'HN': 'Hà Nội', 'Hà Nội.': 'Hà Nội',
    'Đà Nẵng.': 'Đà Nẵng', 'Đà Nẵng': 'Đà Nẵng', 'Cần Thơ.': 'Cần Thơ'
}

major_cities = {
    'Hồ Chí Minh': ['Quận 1', 'Quận 3', 'Quận 5', 'Quận 10', 'Bình Thạnh', 'Phú Nhuận', 'Quận 7'],
    'Hà Nội': ['Ba Đình', 'Hoàn Kiếm', 'Đống Đa', 'Hai Bà Trưng', 'Cầu Giấy', 'Thanh Xuân', 'Tây Hồ']
}

north = ['Hà Nội', 'Hải Phòng', 'Quảng Ninh', 'Bắc Ninh', 'Hưng Yên']
central = ['Đà Nẵng', 'Thừa Thiên Huế', 'Quảng Nam']
south = ['Hồ Chí Minh', 'Cần Thơ', 'Bình Dương', 'Đồng Nai']

def get_region(province):
    if province in north:
        return 'North'
    elif province in central:
        return 'Central'
    elif province in south:
        return 'South'
    return 'Other'

def prepare_input(form_input):
    address = form_input['address']
    province = address.split(',')[-1].strip()
    province = province_mapping.get(province, province)

    # Thành phố lớn và trung tâm
    is_major_city = int(province in major_cities)
    is_center = 0
    if is_major_city:
        for d in major_cities[province]:
            if d in address:
                is_center = 1
                break

    # Region
    region = get_region(province)

    # Các thông tin cơ bản
    area = int(form_input['area'])
    floors = int(form_input['floors'])
    bedrooms = int(form_input['bedrooms'])
    bathrooms = int(form_input['bathrooms'])

    # Optional: cho phép nhập AveragePricePerSquare, nếu không có gán mặc định
    avg_price = float(form_input.get('AveragePricePerSquare', 0))

    # Tạo DataFrame
    df = pd.DataFrame([{
        'Area': area,
        'Frontage': form_input['frontage'],
        'Access Road': form_input['access_road'],
        'Floors': floors,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Legal status': form_input['legal_status'],
        'Furniture state': form_input['furniture_state'],
        'is_major_city': is_major_city,
        'is_center_district': is_center,
        'region': region,
        'Total Room': bedrooms + bathrooms,
        'Total Room + Floor': bedrooms + bathrooms + floors,
        'Bedrooms * Area': bedrooms * area,
        'Bathrooms * Area': bathrooms * area,
        'Area_Floors': area * floors,
        'High_Values Real Estate': is_major_city * is_center,
        'AveragePricePerSquare': avg_price
    }])

    print("\n📌 ✅ Dữ liệu gốc chưa scale:")
    print(df.T)

    # Chạy qua pipeline
    X_ready = preprocessor.transform(df)

    # Chuyển thành DataFrame để dễ đọc log
    X_ready_df = pd.DataFrame(X_ready, columns=feature_list)

    print("\n📏 ✅ Dữ liệu sau khi scale:")
    print(X_ready_df.T)

    return X_ready_df
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        try:
            print(request.form)
        except Exception as e:
            prediction = f"Lỗi: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
