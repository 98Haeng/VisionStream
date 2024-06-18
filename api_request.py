# FastAPI 실행 후, API request를 통해 결과를 확인합니다.
import requests
import base64

# API URL 설정
# 본인의 pc에서는 위와같이
# 타 pc에서 api연결 후 사용한다면, http://접속하려는컴퓨터의ipv4주소:포트번호/
# 211.177.104.197 : 이신행 개인 pc IPv4 주소
# http://211.177.104.197:8000/predict/
url = "http://localhost:8000/predict/"

image_path = 'sbData/test/img1.jpg'
# JSON 요청 본문 구성
data = {
    "image": image_path
}

# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 출력
print("Status Code:", response.status_code)
print("Response Body:", response.json())