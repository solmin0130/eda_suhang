import streamlit as st
import pyrebase
import time
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Firebase 설정
# ---------------------
firebase_config = {
    "apiKey": "AIzaSyCswFmrOGU3FyLYxwbNPTp7hvQxLfTPIZw",
    "authDomain": "sw-projects-49798.firebaseapp.com",
    "databaseURL": "https://sw-projects-49798-default-rtdb.firebaseio.com",
    "projectId": "sw-projects-49798",
    "storageBucket": "sw-projects-49798.firebasestorage.app",
    "messagingSenderId": "812186368395",
    "appId": "1:812186368395:web:be2f7291ce54396209d78e"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
firestore = firebase.database()
storage = firebase.storage()

# ---------------------
# 세션 상태 초기화
# ---------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.id_token = ""
    st.session_state.user_name = ""
    st.session_state.user_gender = "선택 안함"
    st.session_state.user_phone = ""
    st.session_state.profile_image_url = ""


# ---------------------
# 홈 페이지 클래스
# ---------------------
class Home:
    def __init__(self, login_page, register_page, findpw_page):
        st.title("🏠 Home")
        if st.session_state.get("logged_in"):
            st.success(f"{st.session_state.get('user_email')}님 환영합니다.")

        st.markdown("""
                ---
                **Bike Sharing Demand 데이터셋**  
                - 제공처: [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand)  
                - 설명: 2011–2012년 자전거 대여량 기록 데이터

                **Population Trends 데이터셋**  
                - 분석 목적: 연도 및 지역별 인구 변화 및 트렌드 파악
                - 주요 컬럼: `연도`, `지역`, `인구`, `출생아수(명)`, `사망자수(명)`
                """)


# ---------------------
# EDA 페이지 클래스
# ---------------------
class EDA:
    def __init__(self):
        st.title("📊 EDA")
        tabs = st.tabs(["Bike Sharing EDA", "Population Trends"])

        # --- Bike Sharing EDA ---
        with tabs[0]:
            uploaded = st.file_uploader("Bike Sharing - train.csv 업로드", type="csv", key="bike")
            if uploaded:
                df = pd.read_csv(uploaded, parse_dates=['datetime'])
                st.subheader("Sample 데이터")
                st.dataframe(df.head())
            else:
                st.info("train.csv 파일을 업로드 해주세요.")

        # --- Population Trends ---
        with tabs[1]:
            uploaded = st.file_uploader("Population Trends - population_trends.csv 업로드", type="csv", key="pop")
            if uploaded:
                df = pd.read_csv(uploaded)
                df = df.replace('-', 0)
                df[['인구', '출생아수(명)', '사망자수(명)']] = df[['인구', '출생아수(명)', '사망자수(명)']].apply(pd.to_numeric)

                ptabs = st.tabs(["Basic Stats", "Yearly Trends", "By Region", "Change Analysis", "Visualization"])

                with ptabs[0]:
                    st.subheader("1. Data Info")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    st.subheader("2. Descriptive Stats")
                    st.dataframe(df.describe())
                    st.subheader("3. Missing & Duplicates")
                    st.write(df.isnull().sum())
                    st.write(f"Duplicate rows: {df.duplicated().sum()}")

                with ptabs[1]:
                    st.subheader("National Population Trends")
                    nation = df[df['지역'] == '전국']
                    plt.figure(figsize=(10, 4))
                    plt.plot(nation['연도'], nation['인구'], marker='o')
                    plt.title("Population Trend (National)")
                    plt.xlabel("Year")
                    plt.ylabel("Population")

                    recent = nation.tail(3)
                    avg_diff = (recent['출생아수(명)'] - recent['사망자수(명)']).mean()
                    last_year = nation['연도'].max()
                    last_pop = nation['인구'].iloc[-1]
                    est_2035 = last_pop + (2035 - last_year) * avg_diff
                    plt.axhline(est_2035, color='red', linestyle='--')
                    plt.text(2035, est_2035, f"2035: {int(est_2035):,}", color='red')
                    st.pyplot(plt.gcf())

                with ptabs[2]:
                    st.subheader("Recent 5-Year Regional Change")
                    recent_years = df['연도'].max() - 5
                    regional_df = df[df['지역'] != '전국']
                    pivot = regional_df.pivot(index='지역', columns='연도', values='인구')
                    diff = pivot[df['연도'].max()] - pivot[recent_years]
                    diff = diff.sort_values(ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=diff.values / 1000, y=diff.index, ax=ax, palette="Blues_r")
                    for i, v in enumerate(diff.values / 1000):
                        ax.text(v + 0.5, i, f"{int(v):,}", va='center')
                    ax.set_title("Population Change (Last 5 Years)")
                    ax.set_xlabel("Population Change (Thousands)")
                    st.pyplot(fig)

                    rate = (pivot[df['연도'].max()] - pivot[recent_years]) / pivot[recent_years] * 100
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=rate.values, y=rate.index, ax=ax2, palette="coolwarm")
                    ax2.set_title("Population Change Rate (%)")
                    ax2.set_xlabel("Change Rate")
                    st.pyplot(fig2)

                    st.markdown(
                        "> **Interpretation:** Regions with the largest increase/decrease in population over the past 5 years are displayed above."
                    )

                with ptabs[3]:
                    st.subheader("Top 100 Population Change Cases")
                    df_sorted = df[df['지역'] != '전국'].copy()
                    df_sorted['증감'] = df_sorted.groupby('지역')['인구'].diff()
                    top_change = df_sorted.sort_values('증감', ascending=False).head(100)
                    styled = top_change.style.background_gradient(subset=['증감'], cmap='bwr').format({'증감': "{:,}"})
                    st.dataframe(styled)

                with ptabs[4]:
                    st.subheader("Area Chart by Region")
                    pivot = df[df['지역'] != '전국'].pivot(index='연도', columns='지역', values='인구')
                    pivot = pivot.fillna(0)
                    pivot = pivot / 1000
                    fig, ax = plt.subplots(figsize=(12, 6))
                    pivot.plot.area(ax=ax, cmap='tab20')
                    ax.set_title("Stacked Population by Region")
                    ax.set_ylabel("Population (Thousands)")
                    ax.set_xlabel("Year")
                    st.pyplot(fig)


# ---------------------
# 로그인 페이지 클래스
# ---------------------
class Login:
    def __init__(self):
        st.title("🔐 로그인")
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        if st.button("로그인"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.id_token = user['idToken']

                user_info = firestore.child("users").child(email.replace(".", "_")).get().val()
                if user_info:
                    st.session_state.user_name = user_info.get("name", "")
                    st.session_state.user_gender = user_info.get("gender", "선택 안함")
                    st.session_state.user_phone = user_info.get("phone", "")
                    st.session_state.profile_image_url = user_info.get("profile_image_url", "")

                st.success("로그인 성공!")
                time.sleep(1)
                st.rerun()
            except Exception:
                st.error("로그인 실패")


# ---------------------
# 회원가입 페이지 클래스
# ---------------------
class Register:
    def __init__(self, login_page_url):
        st.title("📝 회원가입")
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        name = st.text_input("성명")
        gender = st.selectbox("성별", ["선택 안함", "남성", "여성"])
        phone = st.text_input("휴대전화번호")

        if st.button("회원가입"):
            try:
                auth.create_user_with_email_and_password(email, password)
                firestore.child("users").child(email.replace(".", "_")).set({
                    "email": email,
                    "name": name,
                    "gender": gender,
                    "phone": phone,
                    "role": "user",
                    "profile_image_url": ""
                })
                st.success("회원가입 성공! 로그인 페이지로 이동합니다.")
                time.sleep(1)
                st.switch_page(login_page_url)
            except Exception:
                st.error("회원가입 실패")


# ---------------------
# 비밀번호 찾기 페이지 클래스
# ---------------------
class FindPassword:
    def __init__(self):
        st.title("🔎 비밀번호 찾기")
        email = st.text_input("이메일")
        if st.button("비밀번호 재설정 메일 전송"):
            try:
                auth.send_password_reset_email(email)
                st.success("비밀번호 재설정 이메일을 전송했습니다.")
                time.sleep(1)
                st.rerun()
            except:
                st.error("이메일 전송 실패")


# ---------------------
# 사용자 정보 수정 페이지 클래스
# ---------------------
class UserInfo:
    def __init__(self):
        st.title("👤 사용자 정보")

        email = st.session_state.get("user_email", "")
        new_email = st.text_input("이메일", value=email)
        name = st.text_input("성명", value=st.session_state.get("user_name", ""))
        gender = st.selectbox(
            "성별",
            ["선택 안함", "남성", "여성"],
            index=["선택 안함", "남성", "여성"].index(st.session_state.get("user_gender", "선택 안함"))
        )
        phone = st.text_input("휴대전화번호", value=st.session_state.get("user_phone", ""))

        uploaded_file = st.file_uploader("프로필 이미지 업로드", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_path = f"profiles/{email.replace('.', '_')}.jpg"
            storage.child(file_path).put(uploaded_file, st.session_state.id_token)
            image_url = storage.child(file_path).get_url(st.session_state.id_token)
            st.session_state.profile_image_url = image_url
            st.image(image_url, width=150)
        elif st.session_state.get("profile_image_url"):
            st.image(st.session_state.profile_image_url, width=150)

        if st.button("수정"):
            st.session_state.user_email = new_email
            st.session_state.user_name = name
            st.session_state.user_gender = gender
            st.session_state.user_phone = phone

            firestore.child("users").child(new_email.replace(".", "_")).update({
                "email": new_email,
                "name": name,
                "gender": gender,
                "phone": phone,
                "profile_image_url": st.session_state.get("profile_image_url", "")
            })

            st.success("사용자 정보가 저장되었습니다.")
            time.sleep(1)
            st.rerun()


# ---------------------
# 로그아웃 페이지 클래스
# ---------------------
class Logout:
    def __init__(self):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.id_token = ""
        st.session_state.user_name = ""
        st.session_state.user_gender = "선택 안함"
        st.session_state.user_phone = ""
        st.session_state.profile_image_url = ""
        st.success("로그아웃 되었습니다.")
        time.sleep(1)
        st.rerun()


# ---------------------
# 페이지 객체 생성 및 실행
# ---------------------
Page_Login = st.Page(Login, title="Login", icon="🔐", url_path="login")
Page_Register = st.Page(lambda: Register(Page_Login.url_path), title="Register", icon="📝", url_path="register")
Page_FindPW = st.Page(FindPassword, title="Find PW", icon="🔎", url_path="find-password")
Page_Home = st.Page(lambda: Home(Page_Login, Page_Register, Page_FindPW), title="Home", icon="🏠", url_path="home",
                    default=True)
Page_User = st.Page(UserInfo, title="My Info", icon="👤", url_path="user-info")
Page_Logout = st.Page(Logout, title="Logout", icon="🔓", url_path="logout")
Page_EDA = st.Page(EDA, title="EDA", icon="📊", url_path="eda")

#-----------------
# 내비게이션 실행
#-----------------
if st.session_state.logged_in:
    pages = [Page_Home, Page_User, Page_Logout, Page_EDA]
else:
    pages = [Page_Home, Page_Login, Page_Register, Page_FindPW]

selected_page = st.navigation(pages)
selected_page.run()