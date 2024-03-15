import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, render_template
from data import player_table

app = Flask(__name__)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# BERT 모델 불러오기
model_path = 'bert_pt_model.pth'
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # 모델을 평가 모드로 설정

# 선수/팀 BERT 수행
def select_pt(user_question):
    # 텍스트를 리스트로 변환합니다.
    test_texts = list(user_question)
    # 입력 문장을 토크나이징하고 텐서 데이터셋을 생성합니다.
    test_encodings = tokenizer(user_question, truncation=True, padding=True)
    test_dataset = TensorDataset(torch.tensor([test_encodings['input_ids']]), torch.tensor([test_encodings['attention_mask']]))
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 모델을 평가 모드로 설정합니다.
    model.eval()
    tp_prediction = []
    with torch.no_grad():
        for input_ids, attention_mask in test_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            tp_prediction.extend(preds.cpu().numpy())

    return tp_prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.form['user_question']
    result_pt = select_pt(user_question)

    # 선수/팀 BERT의 결과를 이용하여 응답 생성
    if result_pt[0] == 0:
        result = '''선수 정보가 선택되었습니다. 카테고리를 선택해 주세요.<br>
        <br>
        - 선수 등록번호<br>
        - 선수 기본정보 검색<br>
        - 선수 전체 기록 검색<br>
        - 선수별 골 예측<br>
        - 선수별 도움 예측<br> 
        '''
    else:
        result = '''팀 정보가 선택되었습니다. 카테고리를 선택해 주세요.<br>
        <br>
        - 구장 지도 링크<br>
        - 틸별 승점 및 순위 예측(K리그1)<br>
        - 팀별 승점 및 순위 예측(K리그2)<br>
        '''

    return jsonify({'response': result})

@app.route('/test', methods=['POST'])
def player_table():
    user_question = request.form['user_question']

    # 선수/팀 BERT의 결과를 이용하여 응답 생성
    if user_question == '선수 등록번호':
        result = "선수 등록번호를 검색합니다."
    else:
        result = "테스트 실패"

    return jsonify({'response': result})
    
if __name__ == '__main__': 
    app.run(debug=True)

    






#     @app.route('/chat', methods=['POST'])
# def chat():
#     user_question = request.form['user_question']
#     if "선수 정보" in user_question:
#         response = "선수 정보가 선택되었습니다. 선수 이름을 입력해주세요."
#     elif "팀 정보" in user_question:
#         response = "팀 정보가 선택되었습니다. 팀 이름을 입력해주세요."
#     else:
#         response = "선수 정보 또는 팀 정보를 선택해주세요."
#     return jsonify({'response': response})

# @app.route('/name', methods=['POST'])
# def handle_name():
#     name = request.form['name']
#     response = f"{name}에 대한 정보를 조회합니다. 어떤 정보를 원하시나요? 1. 기본 정보 2. 전적 3. 경기 기록"
#     return jsonify({'response': response})

# @app.route('/player-info', methods=['POST'])
# def player_info():
#     choice = request.form['choice']
#     df = pd.read_excel('player_data_2013-2023.xlsx', engine='openpyxl')
#     headers = df.columns.tolist()
#     response = f"선택하신 옵션: {choice}\n엑셀 파일 헤더: {headers}"
#     return jsonify({'response': response})