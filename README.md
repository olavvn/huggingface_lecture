# 허깅페이스 익히기

<aside>
💡 **학습 목표
허깅페이스에 대해 배우고, 사용법을 익혀보자.**

---

</aside>

### 0. 구글 코랩(Google Colab)과 깁헙(Github) 연동하기

[구글 코랩(Google Colab)과 깃헙(Github)연동하기](https://www.notion.so/Google-Colab-Github-1e5219be4e0b8034ab55f32271c03a31?pvs=21) 

# 1. 허깅페이스 개요

## 1.1. 등장 배경

- 문제점: 쏟아져 나오는 트랜스포머 아키텍쳐, but 구현 방식에 차이가 존재.
    
    → 모델마다 활용법을 익혀야 하는 어려움
    
- 해결 방안: 공통된 인터페이스로 트랜스포머 모델을 활용할 수 있도록 지원하는 라이브러리 개발
    
    → 허깅페이스(Huggingface) 팀의 트랜스포머 라이브러리
    

## 1.2. 허깅페이스란?

- 다양한 트랜스포머 모델을 통일된 인터페이스로 사용 가능하도록 지원하는 오픈소스 라이브러리/플랫폼
    - 라이브러리
        - `transformers` 사전 학습된 트랜스포머 모델 및 토크나이저 지원.
        - `datasets` 데이터셋 업로드 및 다운로드 지원.
        - 서로 다른 모델을 통일된 인터페이스로 활용할 수 있도록 함(실습1)
        
        ```python
        text = "What is Huggingface Transformers?"
        # BERT 모델 활용
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        encoded_input = bert_tokenizer(text, return_tensors='pt')
        bert_output = bert_model(**encoded_input)
        # GPT-2 모델 활용
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        encoded_input = gpt_tokenizer(text, return_tensors='pt')
        gpt_output = gpt_model(**encoded_input)
        ```
        
    - 플랫폼(Hub)
        - `huggingface_hub` 모델과 데이터셋 탐색 및 공유 지원.
        - https://huggingface.co/
- 새로운 모델 사용법을 익히는 데 필요한 로드를 줄여주어 빠르고 간편한 모델 활용을 가능하게 함
    
    → AI 개발 생태계 개선.
    

## 1.3. 허깅페이스 허브 탐색하기

### 1.3.1. 모델 허브

https://huggingface.co/models

- 필터링
    - `Tasks` 작업 종류에 따른 모델 필터링
        - `NLP`, `CV`, `Audio`, `Multimodal`, etc.
    - `Libraries` 모델 학습에 사용된 라이브러리에 따른 필터링
        - `PyTorch`, `TensorFlow`, `Transformers` , etc.
    - `Datasets` , `Languages` 등 필터링 가능
- 모델 탐색
    - https://huggingface.co/deepseek-ai/DeepSeek-R1
    - 모델 이름 및 요약 정보
    - 모델 설명
    - 모델 트렌드
    - 모델 테스트 API
    
    ![그림1.png](attachment:7df9759c-485a-4533-8732-a8bb0719734b:f2dc067b-aba3-4c24-b754-b5ac0f81bac5.png)
    

### 1.3.2. 데이터셋 허브

https://huggingface.co/datasets

- 필터링에 사이즈와 데이터 형식 추가
- KLUE(Korean Language Understanding Evaluation) Dataset
    - https://huggingface.co/datasets/klue/klue
    - Subset: 데이터셋의 하위 데이터셋
    - Spilt: Train, Validation 8:2 비율

# 2. 트랜스포머 모델 활용하기

## 2.1. 모델 불러오기

### 2.1.1. 바디 + 헤드

- 바디: 모델의 중심 엔진. 추론 모델의 핵심 부분.
- 헤드: 기능에 따라 세분화된 특정 태스크를 수행하는 부분.
- 허깅페이스 라이브러리로 동일한 바디에 서로 다른 헤드를 붙여 다양한 작업을 할 수 있음.

![image.png](attachment:0f918fa1-e9fe-400e-a88f-3fbe155c3f2c:image.png)

![image.png](attachment:c1c1ed53-a22b-4c41-81ae-fefadc4f2049:image.png)

### 2.1.2. 모델 불러오기

1. 모델 아이디로 바디만 불러오기(실습2)
    
    `AutoModel`: 헤드 지정 안한 body
    
    ```python
    from transformers import AutoModel
    base_model_id = 'bert-base-uncased'
    base_model = AutoModel.from_pretrained(base_model_id)
    ```
    
2. 분류 헤드가 포함된 모델 불러오기(실습3)
    
    `AutoModelForSequenceClassification`: 헤드 지정된 모델
    
    ```python
    from transformers import AutoModelForSequenceClassification
    model_id = 'SamLowe/roberta-base-go_emotions'
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    ```
    
3. 분류 헤드가 랜덤으로 초기화된 모델 불러오기(실습4)
    
    ```python
    from transformers import AutoModelForSequenceClassification
    model_id = 'klue/roberta-base'
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    ```
    

## 2.2. 토크나이저 활용하기

### 2.2.1. 토크나이저

- 텍스트를 토큰 단위로 나누고 각 토큰을 대응하는 토큰 아이디로 변환하는 아키텍쳐
    
    토큰: 언어 모델이 텍스트를 이해하고 생성하는 기본 단위
    
- 학습데이터를 통해 토큰 사전 구축→모델마다 토크나이징하는 방법이 다름.
- 동일한 모델 아이디로 모델과 토크나이저를 통일해야 함.
    
    (실습5) 토크나이저 불러오기
    
    ```python
    from transformers import AutoTokenizer
    model_id = 'klue/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ```
    

### 2.2.2. 토크나이저 출력값 및 주요 메서드

- 반환: 토큰화된 문장의 다양한 인덱스
    - input_ids: 토큰화했을 때 각 토큰이 토크나이저 사전의 몇 번째 항목인지 표시
        - input_ids 출력 예시
            
            ```python
            # {'input_ids': [0, 9157, 7461, 2190, 2259, 8509, 2138, 2855, 5385, 2200, 20950, 2],
            # 항상 문장의 시작은 0, 끝은 2.
            ```
            
    - token_type_ids: 토큰이 속한 문장의 아이디. BERT모델에서는 문장 토큰 구분이 필요가 없으므로 모두 0(그냥 알아두자)
    - attention_mask: 해당 토큰이 길이를 맞추기 위한 padding_token일 경우 0, 아닐 경우 1로 세팅.
        - padding_token: 문장 길이를 맞추기 위하여 문장에 덧붙인 토큰으로 input_ids상에서 1로 표시됨.
- 메서드:
    - 실습6. convert_ids_to_tokens: 문장의 토큰화 결과를 리스트로 반환.
        
        ```python
        print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))
        # ['[CLS]', '토크', '##나이', '##저', '##는', '텍스트', '##를', '토', '##큰', '단위', '##로', '나눈다', '[SEP]']
        # 항상 '[CLS]'로 시작, '[SEP]'로 종료.
        ```
        
    - 실습6. decode: 토큰화된 문장을 원래 문장으로 디코딩
        
        ```python
        print(tokenizer.decode(tokenized['input_ids']))
        # [CLS] 토크나이저는 텍스트를 토큰 단위로 나눈다 [SEP]
        print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))
        # 토크나이저는 텍스트를 토큰 단위로 나눈다
        ```
        
    - 실습9. batch_decode: 토큰 리스트의 리스트를 문자열 리스트로 디코딩
        
        ```
        first_tokenized_result = tokenizer(['첫 번째 문장', '두 번째 문장'])['input_ids']
        print(tokenizer.batch_decode(first_tokenized_result))
        # ['[CLS] 첫 번째 문장 [SEP]', '[CLS] 두 번째 문장 [SEP]']
        
        second_tokenized_result = tokenizer([['첫 번째 문장', '두 번째 문장']])['input_ids']
        print(tokenizer.batch_decode(second_tokenized_result))
        # ['[CLS] 첫 번째 문장 [SEP] 두 번째 문장 [SEP]']
        ```
        

# 3. 데이터셋 활용하기

## 3.1. 데이터셋 다운로드

- `load_dataset`: 허깅페이스에서 데이터셋 불러오는 함수. 데이터셋 “이름”과 “서브셋 이름”을 인자로 받아 내려 받고, train, test로 나눠줌(split, 0.8이 기본)
    - 실습12. 허깅페이스 허브에서 데이터셋 다운로드
        
        ```python
        from datasets import load_dataset
        klue_mrc_dataset = load_dataset('klue', 'mrc')
        klue_mrc_dataset_only_train = load_dataset('klue', 'mrc', split='train')
        ```
        
- 허깅페이스 데이터 뿐만 아니라 로컬 데이터로도 비슷한 작업 가능. 이 경우에는 데이터셋을 split하지 않고, 모든 데이터를 train으로 지정함
    - 실습13. 로컬의 데이터 활용하기
    
    ```python
    # 로컬의 데이터 파일을 활용
    dataset_json = load_dataset("json", data_files="/content/drive/MyDrive/data/review_신라스테이 해운대.json")
    #print(dataset_json)
    
    #로컬 데이터로는 train 100%이 default.
    dataset_json_train = load_dataset("json", data_files="/content/drive/MyDrive/data/review_신라스테이 해운대.json", split='train')
    #print(dataset_json_train)
    
    #수동으로 split 하는 방법
    dataset_json_test = dataset_json["train"].train_test_split(test_size=0.2, seed=42)["test"]
    #print(dataset_json_test)
    ```
    
- Python 딕셔너리와 pandas library를 이용하여 데이터셋 제작 가능(실습14).
    
    ```python
    # 파이썬 딕셔너리 활용
    from datasets import Dataset
    my_dict = {"a": [1, 2, 3]}
    dataset = Dataset.from_dict(my_dict)
    print(dataset[0])
    
    # 판다스 데이터프레임 활용
    from datasets import Dataset
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    dataset = Dataset.from_pandas(df)
    print(dataset[0])
    ```
    

## 3.2. 데이터셋 가공하기

KLUE 데이터셋의 YNAT 서브셋을 이용하여 데이터셋 가공 실습을 진행. 이 데이터에서 필요한 정보는 기사 제목과 토픽이라고 가정.

- 데이터셋 확인
    
    ```json
    //train
    Dataset({
        features: ['guid', 'title', 'label', 'url', 'date'],
        num_rows: 45678
    })
    //eval
    Dataset({
        features: ['guid', 'title', 'label', 'url', 'date'],
        num_rows: 9107
    })
    // guid: 데이터의 고유 ID
    ```
    
- 필요 없는 정보 제거.
    - guid, url, date와 같은 정보는 필요 없으므로 제거해야함.
    - remove_columns: 지정한 칼럼을 데이터셋에서 제거하는 메서드
    - 실습15. 실습에 사용하지 않는 불필요한 컬럼 제거
        
        ```python
        klue_tc_train_removed = klue_tc_train.remove_columns(['guid', 'url', 'date'])
        ```
        
- 칼럼의 속성 확인 및 컬럼 추가하기
    - feature: 지정한 컬럼의 속성(Class)을 출력하는 메서드
    
    ```python
    print(klue_tc_train_removed.features['title'])
    print(klue_tc_train_removed.features['label'])
    #Value(dtype='string', id=None)
    #ClassLabel(names=['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치'], id=None)
    ```
    
    - label을 숫자에서 문자열로 바꾸고싶음
    - `int2str`: ID에 해당하는 문자열로 변환해주는 메서드
    - 실습16. 카테고리를 문자로 표기한 label_str 컬럼 추가
    
    ```python
    klue_tc_label = klue_tc_train_removed.features['label']
    
    def make_str_label(batch):
      batch['label_str'] = klue_tc_label.int2str(batch['label'])
      return batch
    #batch: 한번에 처리할 데이터 묶음. For 문 사용하면서 생기는 overhead줄여줌.
    klue_tc_train_removed = klue_tc_train_removed.map(make_str_label, batched=True, batch_size=1000)
    ```
    
- Train/Validation/Test
    - Train: 학습을 위한 데이터, Validation: 학습 중간의 체크포인트(이 결과 보고 optimization 진행), Test: 학습 종료 후 모델 성능 평가(다음시간에..)
    - 실습17. 학습/검증/테스터 데이터셋 분할
        
        ```python
        train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
        dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
        test_dataset = dataset['test']
        valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']
        ```
        

# 4. 모델을 이용하여 추론하기

언어 모델(Language Model): 주어진 단어 시퀀스 다음에 어떤 단어가 나올 확률을 예측하는 모델. 

다양한 아키텍쳐(RNN, CNN)가 있지만, 우리는 어텐션(Attention) 메커니즘의 트랜스포머 아키텍쳐에 관심을 둠.

## 4.1.  BERT Model vs GPT Model

| 특징 | BERT (Bidirectional Encoder Representations from Transformers) | GPT (Generative Pre-trained Transformer) |
| --- | --- | --- |
| **아키텍처** | 트랜스포머의 **인코더(Encoder)** 부분만 사용 | 트랜스포머의 **디코더(Decoder)** 부분만 사용 |
| **방향성** | **양방향(Bidirectional)** | **단방향(Unidirectional, 주로 왼쪽->오른쪽)** |
| **사전 학습 목표** | 1. **MLM (Masked Language Model):** 문장의 일부 단어를 가리고 양쪽 문맥을 보고 예측
2. **NSP (Next Sentence Prediction):** 두 문장이 이어지는지 여부 예측 | **표준 언어 모델링:** 이전 단어(들)를 보고 다음 단어 예측 |
| **주요 강점** | **텍스트 이해 및 분석(**문맥 속 단어 의미, 문장 간 관계 파악 능력 우수) | **텍스트 생성**(자연스럽고 일관성 있는 문장/문단 생성 능력 우수) |
| **주요 활용 분야** | 분류 (스팸 감지, 감성 분석), 개체명 인식, 질의응답 (정답 추출), 검색, 요약 (추출 요약) | 텍스트 생성 (글쓰기, 코딩), 대화형 AI, 번역, 요약 (추상 요약) |
| **활용 방식** | 특정 작업에 맞춘 **Fine-tuning**이 일반적→라벨링된 데이터 | Fine-tuning 외에 **Zero-shot(프롬프트 엔지니어링), Few-shot 학습**으로도 강력한 성능 |
| **정보 처리 방식** | 주어진 전체 문맥 속에서 각 단어의 의미를 파악 | 주어진 이전 문맥을 기반으로 다음 내용을 예측하며 확장 |

## 4.2. BERT Model을 이용하여 추론하기

### 4.2.1. 파이프라인을 활용한 추론

- `pipeline`: 토크나이저와 모델을 결합해 데이터의 전후처리와 모델 추론을 간단하게 수행하는 함수.
    
    실습18. 학습한 모델을 불러와 pipeline을 활용해 텍스트 분류하기.
    
    ```python
    from transformers import pipeline
    
    model_id = "hykiim/roberta-base-klue-ynat-classification"
    
    model_pipeline = pipeline("text-classification", model=model_id)
    
    model_pipeline(test_dataset["title"][:5])
    ```
    
    - `pipeline` 작업 종류
        
        **1. Text Classification (텍스트 분류)**
        
        - 텍스트를 입력받아 미리 정의된 범주 중 하나로 분류.
        - 예: 감정 분석, 스팸 감지, 뉴스 기사 분류
        - 모델 예: `bert-base-uncased`, `roberta-base`
        
        **2. Token Classification (토큰 분류)**
        
        - 텍스트의 각 토큰을 입력받아 개체명 인식, 품사 태깅 등의 작업 수행.
        - 예: 개체명 인식, 품사 태깅, 청크 태깅
        - 모델 예: `dbmdz/bert-large-cased-finetuned-conll03-english`, `xlm-roberta-base`
        
        **3. Text Generation (텍스트 생성)**
        
        - 주어진 프롬프트를 기반으로 새로운 텍스트 생성.
        - 예: 기계 번역, 텍스트 요약, 질문 답변
        - 모델 예: `gpt2`, `t5-base`
        
        **4. Question Answering (질문 답변)**
        
        - 텍스트와 질문을 입력받아 텍스트에서 질문에 대한 답변 추출.
        - 예: 독해, FAQ 챗봇
        - 모델 예: `deepset/roberta-base-squad2`, `distilbert-base-uncased-distilled-squad`
        
        **5. Summarization (요약)**
        
        - 긴 텍스트를 입력받아 짧은 요약 생성.
        - 예: 뉴스 기사 요약, 문서 요약
        - 모델 예: `facebook/bart-large-cnn`, `google/pegasus-xsum`
        
        **6. Translation (번역)**
        
        - 텍스트를 입력받아 다른 언어로 번역.
        - 예: 영어-프랑스어 번역, 한국어-영어 번역
        - 모델 예: `Helsinki-NLP/opus-mt-en-fr`, `facebook/wmt19-en-de`
        
        **7. Feature Extraction (특징 추출)**
        
        - 텍스트를 입력받아 숫자 벡터 형태의 특징 추출.
        - 예: 텍스트 유사도 계산, 텍스트 분류
        - 모델 예: `bert-base-uncased`, `sentence-transformers/all-mpnet-base-v2`
        
        **8. Fill-Mask (마스크 채우기)**
        
        - 텍스트에서 마스크된 토큰 예측.
        - 예: 텍스트 완성, 단어 예측
        - 모델 예: `bert-base-uncased`, `roberta-base`
        
        **9. Zero-Shot Classification (제로샷 분류)**
        
        - 학습 데이터 없이 새로운 범주로 텍스트 분류.
        - 예: 감정 분석, 주제 분류
        - 모델 예: `facebook/bart-large-mnli`, `cross-encoder/nli-distilroberta-base`

### 4.2.2.  파이프라인 사용하지 않고 직접 구현하기.

- 실습19. 커스텀 파이프라인 구현하기
    
    ```python
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    class CustomPipeline:
        def __init__(self, model_id):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)#목적에 맞는 모델 헤드 불러옴
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)#모델과 동일한 토크나이저 불러옴
            self.model.eval() #모델 평가 모드(아직은 몰라도 됨.)
    
        def __call__(self, texts):
            tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)# 입력 텍스트를 토크나이저를 사용하여 토큰화
            # return_tensors="pt": PyTorch 텐서로 반환, padding=True: 패딩 추가, truncation=True: 잘라내기
    
            with torch.no_grad():#기울기 계산 비활성화(옵티마이저 계산 생략)-> 추론 과정에서는 옵티마이징 할 필요 없음.
                outputs = self.model(**tokenized)#**연산자는 딕셔너리를 키워드 인자로 unpacking하여 모델에 입력. 즉 토큰 아이디
                logits = outputs.logits# 모델 출력에서 logits 값을 추출. (분류되지 않은 예측값)
    
            #추론이 정확할 확률 구하기
            probabilities = softmax(logits, dim=-1)
            scores, labels = torch.max(probabilities, dim=-1)
            labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]
    
            return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]
    
    custom_pipeline = CustomPipeline(model_id)
    print(custom_pipeline(test_dataset['title'][:5])) 
    print(test_dataset['label_str'][:5])
    ```
    

---

**편집자:** 김도훈

**참고 자료:** 허정준, LLM을 활용한 실전 AI 애플리케이션 개발, 책만.