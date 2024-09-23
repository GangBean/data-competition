## Linux ubuntu 20.04 error
```
sudo apt-get install libxrender1
sudo apt-get install lzma liblzma-dev libbz2-dev

cd ${python_dir}
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
```

## Data Augmentation
- ChEMBL 34 version 을 사용해 아래 쿼리 결과를 새로운 데이터로 활용 [다운로드 링크](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz)
- 기존 1,952 종의 Molecule 과 동일한 개수의 Molecule을 기준으로 추가된 실험결과 데이터가 존재
``` python
import pandas as pd
import sqlite3

DATABASE_URL = "../data/chembl_34.db"
db = sqlite3.connect(DATABASE_URL)

sql = f'''
SELECT m.chembl_id as 'Molecule ChEMBL ID'
     , a.standard_type as 'Standard Type'
     , a.standard_relation as 'Standard Relation'
     , a.standard_value as 'Standard Value'
     , a.standard_units as 'Standard Units'
     , a.pchembl_value as 'pChEMBL Value'
     , aa.chembl_id as 'Assay ChEMBL ID'
     , t.chembl_id as 'Target ChEMBL ID'
     , t.pref_name as 'Target Name'
     , t.organism as 'Target Organism'
     , t.target_type as 'Target Type'
     , d.chembl_id as 'Document ChEMBL ID'
     , a.standard_value as 'IC50_nM'
     , a.pchembl_value as 'pIC50'
     , c.canonical_smiles as 'Smiles'
  FROM compound_structures c
 INNER JOIN molecule_dictionary m ON m.molregno = c.molregno
 INNER JOIN compound_records cr ON cr.molregno = m.molregno
 INNER JOIN activities a ON a.molregno = m.molregno AND a.record_id = cr.record_id AND a.doc_id = cr.doc_id AND a.standard_type = 'IC50' AND a.standard_relation = '=' AND a.standard_units = 'nM'
 INNER JOIN docs d ON d.doc_id = a.doc_id AND d.doc_type IN ('PUBLICATION', 'PATENT')
 INNER JOIN assays aa ON aa.assay_id = a.assay_id AND aa.doc_id = d.doc_id
 INNER JOIN target_dictionary t ON aa.tid = t.tid AND t.target_type = 'SINGLE PROTEIN' AND t.organism = 'Homo sapiens' AND t.chembl_id = 'CHEMBL3778'
'''
cursor = db.execute(sql)
description = [desc[0] for desc in cursor.description]
df = pd.DataFrame(cursor.fetchall(), columns=description)
df = df[df['pIC50'].isna() == False]
```

## Wrap up
### 대회 개요
[배경]
국내 AI 신약개발 생태계를 활성화하고, 신약 연구자들의 연구 의욕을 고취시키며 신약개발에 참여할 젊은 인재를 발굴하기 위해 제2회 신약개발 AI 경진대회 “JUMP AI 2024”를 개최합니다.

[주제]
IRAK4 IC50 활성 예측 모델 개발

[설명]
IRAK4 IC50에 대한 학습용 데이터 1,952종을 이용해 예측모델을 개발. 개발한 예측모델로 경진용 평가 데이터를 사용하여 예측한 값을 제출

### 배운 점
- sqlite3 로 file DB 사용해봄
### 잘한 점
- 회귀할 수 있는 Target을 IC50와 pIC50 중 선택해서 사용할 수 있었는데, 학습에 사용가능한 데이터의 수가 매우 적었기 때문에 분산이 큰 target을 사용해 모델을 학습시킬 경우, 모델이 회귀해야 할 값의 범위가 커지기 때문에, 값이 큰 target 데이터 포인트에 overfitting 될 확률이 높을 것이라 생각해 분산이 작은 pIC50 을 회귀하는 모델을 만들기로 결정함.
  - 논리: 특정 입력 데이터 포인트의 target이 튀면 loss가 커짐 -> backprop 때 각 node 별 gradient 가 튐 -> overfit 발생
  - public score pIC50 XGB 모델: 0.5303859198 / IC50 XGB 모델: 0.1111111111
- 도메인 지식을 습득 및 활용해 활용가능한 feature들을 생각해내 모델의 성능향상에 기여함.
  - Fingerprint 값이 입력 분자를 구성하는 원자 그룹의 정보라는 것을 이용해, 각 분자-원자 별 임베딩을 학습시켜 원자의 특징을 포착할 수 있도록 했고, 성능 향상됨.
  - public score DNN 모델: 0.5218186937 -> 0.5365998391(2.8% 증가)
- 그나마 코드 템플릿 구조는 잡아둬서, 다음 대회에서도 동일한 형태로 구현할 수 있음
  - feature 를 data frame의 칼럼으로 추가하는 함수를 features 모듈에 작성하고, config 파일에서 on/off 할 수 있도록 해둠.
### 아쉬운 점
- EDA 제대로 안함
  - 부스트캠프 대회 때 진행했던 EDA 방식을 복기해가며 일변량, 다변량 별로 EDA를 진행했어야 하는데.. 그러지 못함
- 베이스라인 모델을 잡을 때 구현이 최대한 간단한 모델로 시작하는 게 좋을 것 같다: XGB 가 학습시간 대비 성능이 좋음. 구현하기도 간편함.
- 이번 대회를 참여하면서 목표를 동메달권 이내나 비슷한 정도로 마무리 하고 싶었는데 471/1630 으로 마감했다.
  - 과정이 만족스러웠다면 결과는 크게 아쉽지 않았을텐데, 이도 저도 아닌 상태로 끝나서 아쉬움이 남는다. 처음 대회를 참가할 때 목표를 과정 중심으로 잡았어야 할 듯.
- 실험 관리가 잘 안 됨. 이건 부스트캠프 대회때도 계속 느끼던 부분인데, 아이디어가 생겼을 때 이걸 정리해둔다고 했는데, 이어서 진행이 잘 안된다. 
  - 뭐가 문제지? 단순히 의지가 부족했던 것일까? 일부분 맞긴 한데, 그래도 전날에 생각한 아이디어를 다음날 구현 안하고 넘어가는건...
  - 우선순위를 관리해야 할 것 같다. 항상 진행하다보면 새로운 생각이 나서 그걸 생각하고 어떻게 할까 정리하다가 이전에 진행하던 것을 마무리 안하고 넘어간 경우가 많은 것 같다.
- 대회를 수행하는 프로세스를 명확하게 정해야 할 것 같다.
  - ``다른 사람들의 방법론``을 먼저 조사해보자.
  - 현재 생각되는 아이디어는 ``Feature Engineering을 최대한 진행 -> 전체 feature set #1 fix -> Modeling 최대한 진행 -> 전체 모델 set #1 fix`` 를 한 사이클로 먼저 수행. ver1 feature set 과 ver1 model set 이 fix 되기 전까진 이 순서 유지. ver1 feature/model set 이 모두 구현 완료되면, 이후엔 feature와 model을 아래 기준에 따라 우선순위를 정해 차례대로 수행
    - 예상 성능 향상 가능성
    - 예상 구현 시간/난이도
    - 관심도
    - 위 3가지 항목에 대해 1~5점(낮음 <-> 높음)으로 점수를 매겨 합산점수가 높은 순위부터 실험 진행
  - 실험 템플릿
    - 개요
    - 목적
    - 진행방향
    - 예상결과
    - 실험결과
### 보완할 점
- 다음 대회 시작전엔 진행 과정에서 꼭 얻어가야 할 것들을 명확하게 정하고 시작해야겠다.
- 무조건 XGB 부터 시작하자. -> 베이스라인 모델을 XGB로 하자 앞으로.
- 노션에 필요한 템플릿 들을 만들자.
  - 실험 템플릿
  - 아이디어 우선순위 테이블
### Competition 방법론 조사
- https://www.kaggle.com/discussions/getting-started/150417
- https://www.kaggle.com/discussions/getting-started/44997
- https://medium.com/@eric.perbos/how-to-win-a-kaggle-competition-in-data-science-via-coursera-part-1-5-592ff4bad624