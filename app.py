from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

# Load model và các thành phần cần thiết
model_path = 'model/ann_model.keras'
scaler = joblib.load('model/scaler.pkl')
pca = joblib.load('model/pca.pkl')
feature_list = joblib.load('model/final_feature_list.pkl')
model = load_model(model_path)

# Danh sách các tỉnh lớn và quận trung tâm (ví dụ)
center_districts_hcm = ['Quận 1', 'Quận 3', 'Quận 5', 'Quận 10', 'Quận 4']
center_districts_hn = ['Hoàn Kiếm', 'Đống Đa', 'Hai Bà Trưng', 'Ba Đình', 'Thanh Xuân', 'Cầu Giấy']
major_cities = {
    'Hồ Chí Minh': 'HCM',
    'Hà Nội': 'Hanoi',
    'Đà Nẵng': 'Danang'
}

# Flask app
app = Flask(__name__)

def preprocess_input(form):
    # Nhập liệu từ form
    address = form['address']
    area = float(form['area'])
    frontage = float(form['frontage'])
    access_road = float(form['access_road'])
    floors = float(form['floors'])
    bedrooms = float(form['bedrooms'])
    bathrooms = float(form['bathrooms'])

    # Tạo DataFrame từ input
    df = pd.DataFrame([{
        'Address': address,
        'Area': area,
        'Frontage': frontage,
        'Access Road': access_road,
        'Floors': floors,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms
    }])

    # Feature mở rộng
    df['Total Room'] = df['Bedrooms'] + df['Bathrooms']
    df['Total Room + Floor'] = df['Total Room'] + df['Floors']
    df['Bedrooms * Area'] = df['Bedrooms'] * df['Area']
    df['Bathrooms * Area'] = df['Bathrooms'] * df['Area']
    df['Area_Floors'] = df['Area'] * df['Floors']

    # Gán giá trị thành phố lớn
    city_found = next((city for city in major_cities if city in address), None)
    df['is_major_city'] = 1 if city_found else 0
    df['major_city_name'] = major_cities.get(city_found, 'Other')

    # Quận trung tâm
    is_center = 0
    if city_found == 'Hồ Chí Minh':
        is_center = any(d in address for d in center_districts_hcm)
    elif city_found == 'Hà Nội':
        is_center = any(d in address for d in center_districts_hn)
    df['is_center_district'] = int(is_center)

    # Vùng miền
    if city_found in ['Hà Nội', 'Hải Phòng', 'Quảng Ninh']:
        df['region'] = 'north'
    elif city_found in ['Đà Nẵng', 'Thừa Thiên Huế', 'Quảng Nam']:
        df['region'] = 'central'
    elif city_found in ['Hồ Chí Minh', 'Cần Thơ', 'Bình Dương']:
        df['region'] = 'south'
    else:
        df['region'] = 'other'

    # High value real estate
    df['High_Values Real Estate'] = df['is_major_city'] * df['is_center_district']

    # One-hot encoding region
    for r in ['north', 'central', 'south', 'other']:
        df[f'region_{r}'] = 1 if df['region'].iloc[0] == r else 0

    # Bỏ cột không cần
    df = df.drop(columns=['region', 'major_city_name', 'Address'])

    # Chuẩn hóa và PCA
    df_scaled = scaler.transform(df)
    df_pca = pca.transform(df_scaled)
    df_pca_df = pd.DataFrame(df_pca[:, :6], columns=[f'PC{i+1}' for i in range(6)])

    # Gộp lại để khớp feature gốc
    final_input = pd.concat([pd.DataFrame(df_scaled, columns=df.columns), df_pca_df], axis=1)

    # Đảm bảo thứ tự cột đúng
    for col in feature_list:
        if col not in final_input.columns:
            final_input[col] = 0
    final_input = final_input[feature_list]

    return final_input

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            user_input = preprocess_input(request.form)
            y_pred = model.predict(user_input)[0][0]
            prediction = f"{y_pred:.2f} tỷ VND"
        except Exception as e:
            prediction = f"Lỗi: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
