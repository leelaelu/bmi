#streamlit_app.py
import streamlit as st
import pickle
import requests
from fastai.learner import load_learner
import pandas as pd
import numpy as np  # numpy를 import해야 함

# Streamlit 제목
st.title("주택 가격 예측 서비스!")

# GitHub Raw 파일 URL과 모델 유형
GITHUB_RAW_URL = "https://github.com/leelaelu/bmi/raw/refs/heads/main/xgb_model.pkl"
MODEL_TYPE = "XGBoost"  # "fastai", "scikit-learn Random Forest", or "XGBoost"
CSV_FILE_URL = "https://github.com/leelaelu/bmi/raw/refs/heads/main/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B3%80%EA%B2%BD%EB%B3%B8%20%EC%B5%9C%EC%A2%85%20%EC%A7%84%EC%A7%9C.csv"

height = 0 # 키
weight = 0 # 몸무게

# GitHub에서 파일 다운로드 및 로드
def download_model(url, output_path="model.pkl"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            file.write(response.content)
        return output_path
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

def load_model(file_path, model_type):
    try:
        if model_type == "fastai":
            return load_learner(file_path)  # Fastai 모델 로드
        else:
            with open(file_path, "rb") as file:
                return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# CSV 파일 읽기
def load_csv_with_encodings(url):
    encodings = ["utf-8", "utf-8-sig", "cp949"]
    for encoding in encodings:
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(url, encoding=encoding)
            st.success(f"CSV file loaded successfully with encoding: {encoding}")
            return df
        except Exception as e:
            continue
    st.error("Failed to load CSV file with supported encodings.")
    return None



# 모델 다운로드 및 로드
downloaded_file = download_model(GITHUB_RAW_URL)
if downloaded_file:
    model = load_model(downloaded_file, MODEL_TYPE)
else:
    model = None

if model is not None:
    st.success("Model loaded successfully!")

# CSV 파일 로드 및 출력
df = load_csv_with_encodings(CSV_FILE_URL)
if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # 사용자 입력 레이아웃 생성
    st.write("### User Input Form")
    col1, col2 = st.columns(2)

    if isinstance(model, dict):  # 모델이 딕셔너리인지 확인
        with col1:
            st.write("**Categorical Features**")
            cat_inputs = {}
            if "cat_names" in model and model["cat_names"]:
                for cat in model["cat_names"]:
                    if cat in df.columns:
                        cat_inputs[cat] = st.selectbox(f"{cat}", options=df[cat].unique())

        with col2:
            st.write("**Continuous Features**")
            cont_inputs = {}
            if "cont_names" in model and model["cont_names"]:
                for cont in model["cont_names"]:
                    if cont in df.columns:
                        cont_inputs[cont] = st.text_input(f"{cont}", value="", placeholder="Enter a number")

    else:
        st.error("The loaded model is not in the expected dictionary format.")



# 예측 버튼 및 결과 출력
prediction = 0
if st.button("Predict"):
    try:
        # 입력 데이터 준비
        input_data = []

        # 범주형 데이터 인코딩
        for cat in model["cat_names"]:  # 메타데이터에서 cat_names 가져오기
            if cat in cat_inputs:
                category = cat_inputs[cat]
                encoded_value = model["categorify_maps"][cat].o2i[category]  # 인코딩된 값 가져오기
                input_data.append(encoded_value)

        # 연속형 데이터 정규화
        i = 0
        for cont in model["cont_names"]:  # 메타데이터에서 cont_names 가져오기
            if cont in cont_inputs:
                raw_value = float(cont_inputs[cont])  # 입력값을 float으로 변환
                if i == 1:
                    height = raw_value # 키 정보 수집
                if i == 2:
                    weight = raw_value # 몸무게 정보 수집
                mean = model["normalize"][cont]["mean"]
                std = model["normalize"][cont]["std"]
                normalized_value = (raw_value - mean) / std  # 정규화 수행
                input_data.append(normalized_value)
                i += 1

        # 예측 수행
        columns = model["cat_names"] + model["cont_names"]  # 열 이름 설정
        input_df = pd.DataFrame([input_data], columns=columns)  # DataFrame으로 변환
        prediction = model["model"].predict(input_df)[0]

        # 결과 출력
        y_name = model.get("y_names", ["Prediction"])[0]
        st.success(f"{y_name} : 주 {prediction:.2f}회")
        bmi = weight / ((height/100) ** 2)
        st.write(f"당신의 BMI 지수는 : {bmi:.2f}")
        st.image("https://t3.daumcdn.net/thumb/R720x0/?fname=http://t1.daumcdn.net/brunch/service/user/hO6/image/l7G5BRSPFosd8YvEqUDnLqx8v8k")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# 예측 결과에 따라 콘텐츠 표시

if prediction < 0.5: # 정상
    st.write("### Prediction Result: Low Price Segment")
    st.image("https://via.placeholder.com/300", caption="Low Segment Image 1")
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # YouTube 썸네일
    st.text("추~~")

elif prediction >= 0.5: # 비만 1단계
    st.write("### Prediction Result: Low Price Segment")
    st.image("https://via.placeholder.com/300", caption="Low Segment Image 1")
    st.video("https://youtu.be/swRNeYw1JkY?si=dlDjPClrWnlYhW37")  # YouTube 썸네일
    st.text("추천ㅇㄴㄹ주 3회입니다.")

elif prediction >= 3.5: # 비만 2단계
    st.write("### Prediction Result: Low Price Segment")
    st.image("https://via.placeholder.com/300", caption="Low Segment Image 1")
    st.video("https://youtu.be/_ffhxHV630A?si=7CP9-pcRkZTrLQDv")  # YouTube 썸네일
    st.text("추천 ㄴㅁㅇㄹ 주 4회입니다.")

elif prediction >= 4.5: # 비만 3단계
    st.write("### Prediction Result: Low Price Segment")
    st.image("https://via.placeholder.com/300", caption="Low Segment Image 1")
    st.video("https://youtu.be/LXTW1Nm_Z3Y?si=AFO2-AtPMEUSb5_O")  # YouTube 썸네일
    st.text("추천 운ㅇㄹ주 5회입니다.")
