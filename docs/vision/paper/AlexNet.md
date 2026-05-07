---
layout: default
title: "AlexNet"
permalink: /vision/paper/
#subtitle: 
use_math: true
parent: paper
grand_parent: vision
---


### research questions
- 대규모 이미지 데이터셋에서 deep convolutional neural network 이 기존 방법들보다 classify performance가 더 좋은가?
- gpu를 활용한 대규모 학습이 실제로 dnn을 가능하게 하는가? 대규모 학습을 위해서 어떠한 기술들이 필요한가?
- overfitting을 방지하면서 대규모 모델을 학습할 수 있는 방법은?

### related works
- SIFT, HOG 등의 hand-crafted feature + shallow classifier
- but alexnet: end-to-end deep CNN이 이미지 분류 태스크를 해결할 수 있는지를 확인 (raw pixel에서 deep cnn으로 연결짓고 여기서 feature learning과 classification을 가져가는 것)
- 즉, feature extraction 자체를 네트워크가 학습하고 이걸 기반으로 classification을 진행할 수 있도록 하는 것 (인간과 좀 더 맞닿도록)

### experiments
- 사용한 데이터셋: ImageNet
- structure
    ![[Pasted image 20260312203223.png]]
    - 8-layer deep CNN (5 conv layers, 3 fully connected layers, softmax)
    - 두 개의 gpu를 활용 (parallelization scheme)
        - GPU1가 일부 feature map을 담당하고 GPU2가 나머지 feature map을 담당하는 방식으로 진행
        - 특정 레이어에서만 두 gpu가 연결됨.
    - ReLU activation
        - 기존에는 tanh와 sigmoid를 많이 활용했으나 여기서는 ReLU를 이용함
        - → gradient saturation이 감소하기 때문에 학습 속도가 빨라짐 (실제로 tanh network보다 약 6배 가량 빠르게 학습할 수 있는 능력)
    - LRN (local response normalization) → biological lateral inhibition에서 유래) → feature competition을 통해서 generalization이 향상됨
        - 큰 activation을 가진 뉴런들이 다른 뉴런을 억제하는 방식으로 진행
        - 실제 뉴런에서 나타내는 lateral inhibition을 표방하는 것
    - max pooling (local pooling) → translation invariacne를 확보하고 feature을 downsampling 가능
    - Dropout(fully connected layer에서 이용함)
        - 학습 시 뉴런을 1/2 확률로 제거함 → co-adaption이 감소하고 generalization이 향상됨
    - data augmentation
        - random cropping : 256x256 이미지에서 224x224 로 크롭함.
        - horizontal flipping
        - rgb channel shifting : RGB color intensity perturbation을 추가함

### results
- ILSVRC 2012 classification
    - top-5 error에 대해 26%의 오류를 보이는 기존 sota 모델에 비해 alexnet은 15.3%의 오류율
- conv layer은 초기 → 후반 레이어로 갈수록 점점 복잡한 feature을 학습해나가는 것을 확인할 수 있었음
    - 초기 레이어 : edge, color. 중간 레이어 : texture, patterns 등. 후반 레이어: object parts 등
    - 특히 첫 Layer filter은 Gabor-like filters와 유사한 패턴을 보임 .. V1 receptive field와 유사한 구조라는 점에서 흥미로움 (low-level visual feature)

### discussion
- Deep CNN이 가능하다는 점을 확인하고, 특히 representation learning이 중요하다는 것을 파악함 (feature을 사람이 설계하는 대신 모델이 직접 학습하도록 하는 접근이 가능하다는 것)
- large dataset + large neural network + GPU computation이 합쳐질 때 성능이 매우 향상될 것
- limitation : 계산 비용, 모델 크기 등. 또한 attention이나 recurrence, explicit structure 등의 고급 요소들은 포함되지 않은 feedforward CNN임

### contribution
- 대규모 cnn의 성능 향상
- ReLU, dropout, data augmentation 등의 학습 기술들을 결합한 것
- low-level visual 특징들을 일부 반영하고 있다는 점에서 human-like로의 인사이트 및 발전 가능성