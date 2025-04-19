import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
# Load mô hình và scaler
scaler = joblib.load("model/scaler.pkl")
final_feature_list = joblib.load("model/final_feature_list.pkl")
pca = joblib.load("model/pca.pkl")
model_path = 'model/ann_model.keras'
scaler = joblib.load('model/scaler.pkl')
pca = joblib.load('model/pca.pkl')
feature_list = joblib.load('model/final_feature_list.pkl')
model = load_model(model_path)


# Các quận trung tâm và thành phố lớn
center_districts_hcm = ['Quận 1', 'Quận 3', 'Quận 5', 'Quận 10', 'Quận 4']
center_districts_hn = ['Hoàn Kiếm', 'Đống Đa', 'Hai Bà Trưng', 'Ba Đình', 'Thanh Xuân', 'Cầu Giấy']
major_cities = ['Hồ Chí Minh', 'Hà Nội', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng', 'Khánh Hòa', 'Quảng Ninh', 'Bình Dương']

# Hàm xác định quận trung tâm
def is_center_district(address, city):
    if city == "Hồ Chí Minh":
        return int(any(d in address for d in center_districts_hcm))
    elif city == "Hà Nội":
        return int(any(d in address for d in center_districts_hn))
    return 0

# Hàm chuẩn hóa dữ liệu đầu vào
def prepare_input_data(form_input):
    """
    form_input là dict chứa các trường như:
    {
        'address': "Quận 1, Hồ Chí Minh",
        'area': 65,
        'floors': 3,
        'bedrooms': 3,
        'bathrooms': 2,
        'frontage': 4.5,
        'access_road': 5.0,
        'legal_status': 'Sổ hồng',
        'furniture_state': 'Đầy đủ'
    }
    """

    address = form_input['address']
    province = address.split(',')[-1].strip()
    
    city_category = province if province in major_cities else "Other"
    is_major_city = int(city_category in ['Hồ Chí Minh', 'Hà Nội'])
    is_center = is_center_district(address, city_category)

    df = pd.DataFrame([{
        'Area': form_input['area'],
        'Frontage': form_input['frontage'],
        'Access Road': form_input['access_road'],
        'Floors': form_input['floors'],
        'Bedrooms': form_input['bedrooms'],
        'Bathrooms': form_input['bathrooms'],
        'Legal status': form_input['legal_status'],
        'Furniture state': form_input['furniture_state'],
        'is_major_city': is_major_city,
        'major_city_name': city_category,
        'is_center_district': is_center,
        'region': '',  # Bạn có thể bổ sung thêm nếu cần
        'Total Room': form_input['bedrooms'] + form_input['bathrooms'],
        'Total Room + Floor': form_input['bedrooms'] + form_input['bathrooms'] + form_input['floors'],
        'Bedrooms * Area': form_input['bedrooms'] * form_input['area'],
        'Bathrooms * Area': form_input['bathrooms'] * form_input['area'],
        'Area_Floors': form_input['area'] * form_input['floors'],
        'Province_Project': province + '_' + 'Không dự án',  # placeholder
        'AveragePricePerSquare': 0,  # nếu chưa có dữ liệu thật
        'High_Values Real Estate': is_major_city * is_center
    }])

    # One-hot encoding cho Legal status và Furniture state
    df = pd.get_dummies(df)

    # Thêm các cột còn thiếu để khớp với model input
    for col in final_feature_list:
        if col not in df.columns:
            df[col] = 0

    # Đảm bảo đúng thứ tự feature
    df = df[final_feature_list]

    # Chuẩn hóa
    df_scaled = scaler.transform(df)

    return df_scaled
test_input = {
    'address': 'Quận 1, Hồ Chí Minh',
    'area': 65,
    'floors': 3,
    'bedrooms': 2,
    'bathrooms': 2,
    'frontage': 5.0,
    'access_road': 6.0,
    'legal_status': 'Sổ hồng',
    'furniture_state': 'Đầy đủ'
}

X_ready = prepare_input_data(test_input)
prediction = model.predict(X_ready)
