from flask import Flask, render_template, request
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import requests
app = Flask(__name__)

# Load model và scaler
model = load_model('model/ann_model.keras')
scaler = joblib.load('scaler.pkl')  # scaler đã lưu từ quá trình huấn luyện

# Danh sách các trường đầu vào (27 features)
input_features = [
    'Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
    'is_major_city', 'is_center_district', 'Total Room',
    'Total Room + Floor', 'Bedrooms * Area', 'Bathrooms * Area', 'Area_Floors',
    'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',  # PCA features
    'High_Values Real Estate', 'Province_Project_Hà Nội_1',
    'Province_Project_Hồ Chí Minh_1', 'Province_Project_Đà Nẵng_1',
    'Province_Project_Bình Dương_1', 'Province_Project_Hải Phòng_1',
    'Province_Project_Khánh Hòa_1', 'Province_Project_Quảng Ninh_1'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = []
        for feature in input_features:
            value = float(request.form.get(feature, 0))
            input_data.append(value)

        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0][0]
        return render_template('index.html', prediction=round(prediction, 2))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
