# 1. AWS CLI(Window용) 다운로드 받아 설치 
    https://awscli.amazonaws.com/AWSCLIV2.msi

## aws cli 설치주소
    https://aws.amazon.com/ko/cli/?pg=developertools

# 2. aws 설정
## 윈도우 - cmd --> aws configure
## AWS 엑세스 키 : 
## AWS 시크릿 키 : 
## AWS 리전: ap-northeast-2
## Default output format [txt]: text


# 3. aws 프로젝트 열기
## 다운로드 받은 aws.zip 파일의 압축을 해제
## Pycharm 에서 [File]-[Open] 한 후 aws 디렉토리를 선택하여 프로젝트를 열기


# 4. 프로젝트 실행
## Pycharm 왼쪽 하단 도구 아이콘 중에서 네 번째(Terminal) 을 열기
## (.venv) PS D:\kim2\python\aws>cd aws
## (.venv) PS D:\kim2\python\aws\aws> python app.py runserver
## * Running on http://127.0.0.1:5001     인터넷 주소를 클릭
## * Running on http://192.168.86.218:5001  인터넷 주소를 클릭


# rekognition docs
https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/rekognition/README.md
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition/client/detect_labels.html#