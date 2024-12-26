# 운전자 이상 행동 분류

## 프로젝트 배경 및 목표
24-1 기계학습 수업에서 진행한 '운전자 이상 행동 분류' 프로젝트를 발전시키기 위해 진행한 프로젝트이다. 이전 프로젝트에서는 단순히 pretrained ResNet50 모델을 사용해 운전자의 이상 행동을 분류했으나, 이번 프로젝트에서는 다양한 최적화 기법과 모델 구조 개선 방법을 적용한 ResNet50 모델과 YOLOv5를 사용하여 운전자의 이상 행동을 분류하고자 한다.

## 진행 기간
2024년 11월 14일 ~ 2024년 12월 14일

## 데이터 수집 방법 및 데이터 설명
AL Hub의 '졸음운전 예방을 위한 운전자 상태 정보 영상' 데이터(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=173) 를 사용한다. 이 데이터는 근적외선 카메라로 촬영된 데이터로서 실제 환경, 준 통제 환경, 통제 환경 등 세 가지 환경에서 1,000명의 운전자로부터 총 400시간의 동영상을 취득하고, 이로부터 추출한 355,000장의 이미지를 가공한 데이터셋이다. 

본 프로젝트에서는 LAB 시뮬레이터 환경에서 개인 운전자 250명을 대상으로 125시간의 동영상을 수집하여 112,500장의 이미지를 가공한 통제 환경 데이터셋을 사용하였다. 해당 데이터셋에는 가공 이미지와 가공 이미지 정보가 담긴 JSON 파일 포함되어 있다. 통제 환경 데이터는 실차 환경 주행 중 사고 위험으로 인해 재현하지 못한 졸음 및 부주의(흡연, 통화) 상황이 재현돼 있고 실제 주행에서 발생할 수 있는 다양한 광원 소스가 고려되어 있어 데이터셋의 다양성 및 정밀성 부분에서 우수하다.

또한 본 프로젝트에서는 데이터 저장 공간, 학습 속도, colab에서 제공하는 일일 무료 리소스 허용량 등을 고려하여 통제 환경 데이터 중 validation set(12,560장)만 사용하였다. 


## 문제 해결 방법
### 모델 선택
이전 프로젝트에서는 운전자의 상태를 실시간으로 분석하기 위해 모델의 깊이와 복잡성에 비해 계산 효율이 높은 ResNet50 모델을 사용하였다. 본 프로젝트에서도 같은 이유로 ResNet50 모델을 사용한다. 

운전자의 행동을 정확히 분석하기 위해서는 담배, 휴대전화, 눈, 입과 같은 세부적인 물체를 검출하는 것이 중요하다. YOLO는 객체 검출에 특화된 모델로, 이미지 내에서 특정 객체를 실시간으로 정확히 파악할 수 있어, 운전자의 얼굴 부위를 정확하게 추출하고 이를 기반으로 행동을 분석하는 데 유리하다. 따라서, 정확한 객체 검출을 위해 YOLOv5 역시 사용한다.

### 문제 해결 전략
YOLOv5를 사용하여 운전자의 담배, 휴대전화, 양쪽 눈, 입을 검출한다. 검출된 눈과 입의 상태(개폐 여부)는 ResNet50 모델을 사용하여 판단한다. 

최종 추론 단계에서는 YOLO의 검출 결과를 먼저 확인한다. 확인 결과 만약 담배나 휴대전화가 존재한다면 해당 이미지를 흡연 혹은 통화로 분류한다. 만약 담배와 휴대전화가 포함돼 있지 않다면, ResNet 모델의 판단 결과 중 입의 개폐 여부를 먼저 확인한다. 입이 닫혀 있다면(하품 상태가 아님) 그 다음으로 눈의 개폐 여부를 확인한다. 

### 최종 데이터 전처리
**YOLOv5**
   + **데이터 형식 변경**: AI Hub dataset에 포함된 json 파일을 yolo 훈련에 사용하기 위해 txt 형식으로 변환
   + **데이터 분할**: 전체 이미지를 6:2:2 비율로 나누어 train set, validation set, test set을 생성

**ResNet50**
   + **데이터 라벨링**: json 파일을 확인하여 모든 이미지에서 양쪽 눈과 입 사진을 잘라 파일명을 수정하여 저장
      + 기존 파일명_클래스 이름_Open
      + 기존 파일명_클래스 이름_Close
   + **데이터 언더샘플링**: Open 및 Close 클래스에 포함된 눈과 입 사진의 비율을 조정  
      + close class = closed eye 2250장 + closed mouth 1750장
      + open class = opened eye 2250장 + opened mouth 1750장
   + **데이터 분할**: 전체 이미지를 6:2:2 비율로 나누어 train set, validation set, test set을 생성
   + **데이터 증강**: RandomRotation, RandomVerticalFlip, GaussianBlur적용


## 최종 모델 수정 및 최적화 기법
**YOLOv5**
+ **detect.py 수정**
   + 담배와 휴대전화 유무를 True/False 값으로 return 하도록 수정
   + 양쪽 눈과 입의 confidence 값이 가장 높은 bbox 좌표를 return 하도록 수정
     
**ResNet50**
+ Compound scaling: resnet50 성능 향상을 위해 적용
+ He initialization: 학습 안정성을 위해 수정된 Conv2d layer에 적용
+ dilated convolution: compound scaling을 적용하며 추가된 layer에 대해 dilated convolution을 적용
+ NAdam optimizer: 학습 초반부에 빠르게 학습할 수 있도록 적용
+ CosineAnnealingLR: 학습 후반부에 세밀하게 학습할 수 있도록 적용
+ lr2 규제: 모델의 일반화 성능을 향상시키기 위해 적용
   
**사용하지 않은 전략**
   + ESRGAN: 저화질 데이터를 개선하고자 시도하였지만 격자 모양 아티팩트가 강조되어 사용하지 않음
   + SE block: channel 간 중요도 학습을 위해 추가하였지만 성능이 저하되어 사용하지 않음

## 모델 평가 
**Customized ResNet50 + YOLOv5 (lr2 규제와 GaussianBlur 적용 X)**
+ 개별 모델 Test 결과
   + YOLOv5 mAP50: 98.6%
   + ResNet50 Accuracy: 95.88%
+ 최종 Inference 결과 (ResNet50 + YOLOv5)
  + Accuracy: 88.6%
  + 하품 Class recall: 97%
  + 통화 Class recall: 99%
  + 정상 Class recall: 85%
  + 졸음 Class recall: 100%
  + 흡연 Class recall: 83%
 
**최종 Customized ResNet50 + YOLOv5 (lr2 규제와 GaussianBlur 적용 O)**
+ 개별 모델 Test 결과
   + YOLOv5 mAP50: 98.6%
   + ResNet50 Accuracy: 95.56%
+ 최종 Inference 결과 (ResNet50 + YOLOv5)
  + Accuracy: 91.9%
  + 하품 Class recall: 97%
  + 통화 Class recall: 99%
  + 정상 Class recall: 96%
  + 졸음 Class recall: 100%
  + 흡연 Class recall: 85%

