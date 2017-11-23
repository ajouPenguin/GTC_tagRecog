# GTC_tagRecog
Samsung GTC assignment & SW capstone design projects

## Pain points
넓은 공장에 있는 전산화되지 않은 상품들을 빠르게 전산화하고 관리할 수 없을까?
## Solutions
- 보유한 영상 장치를 이용해 영상을 스캔
- 상자에 붙어있는 Tag 이용
- 원하는 상품을 전산화 및 관리
## Target
전산화하지 않고 인식할 수 있는 tag가 붙은 상품이 재고로 있는 큰 공장
## Stake holders
1. 공장
2. ...
## Differentiation
1. 영상을 저장할 수 있고 간단한 processing unit을 가진 machine을 이용해 재고의 빠르고 쉬운 전산화가 가능
2. 간단한 머신러닝 기법(SVM)을 이용해 공장·회사마다 다른 tag를 가져도 학습을 통해 적용가능
3. ...
## Benefits
- 공장 : 사람의 손을 거치지 않아도 영상만 확보된다면 빠르고 정확하게 재고의 전산화가 가능
## Requirements
- NumPy
- opencv3.3
- python3.x
- scikit-learn (SVM)
- scikit-image
## References
- hog feature : http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
- selective search : https://github.com/AlpacaDB/selectivesearch
