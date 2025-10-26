---
layout: default
title: "[VWFA, attention] 멍 때리다가 글자가 자동으로 인식될 때" 
permalink: /study/VMFA-DMN
parent: study 
---

수업시간에 멍을 때리고 있다가 갑자기 글씨가 명확하게 인식이 되는 순간이 있다. 

글자가 그냥 그림처럼 보이다가 갑자기 단어로 잘 읽혀지는 순간이 있으면서 멍이 깨지곤(?) 한다. 멍 때리는 것이 먼저 멈춘 것인지는 모르겠지만.

이렇게 글자를 묶어서 인식하는 건 뇌의 어느 부위가 관장하는 것일까?  

---

## Visual World Form Area, VWFA

멍을 때리다가 갑자기 글자가 묶음으로 보이는 현상은 뇌의 left fusiform gyrus에 위치한 시각 단어 형태 영역, 즉 VWPA 이 담당한다. 뇌의 ‘우편함’이라고 불리기도 한다. 

https://pmc.ncbi.nlm.nih.gov/articles/PMC2989180/

시각적으로 인식된 단어는 사실 매우 복잡한 그림일 테지만, 우리는 아주 빠르고 자동적으로 그 단어를 읽을 수 있다. (이러한 연구들은 사실 scene 등에서도 이루어진 것으로 알고 있다.) 그리고 이 작업은 VWFA에서 일어나는데, 이 영역이 정확히 어떤 과정을 수행하는 지에 대해 연구한 논문이다.

결과적으로 VWFA에서는 단순한 시각적 패턴이 아니라, 우리에게 언어적으로 익숙한 철자 구조 (statistical regularities of letter) 에 반응한다. 그리고 여느 인지 구조처럼 hierarchical organization을 따르는데, 이 또한 시각 시스템의 ‘학습’된 결과물이라는 사실을 뒷받침한다. 

### neuronal recycling hypothesis

언어적으로 익숙한 ‘철자 구조’라는 데에서, VWFA는 ‘학습’과 뗄래야 뗄 수 없는 것임을 알 수 있다. 

https://pubmed.ncbi.nlm.nih.gov/17964253/

이 연구에 따르면, VWFA는 neuronal recycling hypothesis를 통해 설명된다. 이는 인간이 진화해오면서 가지고 있는 오래된 뇌 회로를 recycle하여 새로운 기능을 수행할 수 있도록 한다는 이론이다. 단어들을 읽고 본인에게 익숙해질 수 있는 정도로 학습하는 과정에서 VWFA는 측두엽 하부 시각 경로 (원래는 FFA, LOC 등 얼굴이나 사물 인식에 사용되는 부분임) 중 일부를 가지고 recycle하여 문자 인식에 특화시킨다는 것이다. 

### Hierarchy Organizaiton

https://pubmed.ncbi.nlm.nih.gov/17610823/

VWFA 내부에 hierarchical 구조가 있다는 것을 밝혀낸 논문이다. 일반적으로 simple → complex 한 정보를 처리하는 방향인 posterior-to-anterior 구조에서 이 단계적인 처리가 이뤄진다는 것을 밝혀내었다. 

- 개별 문자 탐지 (posterior) → bigram (두 글자 조합 처리) → quadrigram (네 글자 조합 처리) → 실제 단어 인식 (anterior)

---

## Attention Network

멍을 때리다가 갑자기 글자가 인식되는 현상은 attention network가 동적으로 전환되었기 때문이라고 할 수 있다. 

뇌에는 크게 다음 두 가지 network들이 있다. 

**Dorsal Attention Network** 

상두정엽과 전두안구영역으로 구성된다. 

top-down 목표 지향적 주의를 담당한다. 단어를 읽을 때 글자의 공간적 위치와 순서를 처리하는 데에 중요한 역할을 한다. 

**Ventral Attention Network** 

측두-두정 접합부 (TPJ)와 하전두회로 구성된다. 

예상치 못한 중요한 자극에 대한 reorienting을 담당하고 있고, pop-out되는 자극에 대해 자동적인 주의를 포착하는 역할을 한다. 

멍을 때리는 것을 mind wandering이라고 한다. 이 상태는 주로 Default Mode Network (DMN)과 관련이 있다. 보통 이 부분은 뇌가 쉴 때 활성화되곤 한다. 

---

## Feature Integration Theory

**preattentive stage** 

기본적인 시각 특질들이 자연스럽게 병렬적으로 처리되는 단계로, 주의를 기울이지 않아도 처리가 가능한 상태이다. 

**focused attention stage** 

개별 특징들이 spatial attention을 통해 결합되는 상태이며, 이때 두정엽이 핵심적인 역할을 한다. 

---

## 정리

멍을 때리고 있을 때 DMN은 멍 때림 상태를 유지하지만 기본적인 읽기 관련 능력은 부분적 활성화를 유지하기 때문에 글자의 기본적인 특징들은 이미 preattentive stage에서 처리되었지만, focused attention이 내부 사고 (dorsal)에 집중되어 통합되지 않은 상태였다. 그러다가 갑자기 attention이 외부 자극 (ventral) 으로 전환되면서, 글자들이 묶음으로 (VWFA) pop-out되면서 (ventral attention network) 인식되는 것이다.