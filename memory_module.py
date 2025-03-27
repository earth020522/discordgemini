# memory_module.py
import json
import os
from datetime import datetime
import tensorflow_hub as hub 
import numpy as np 
import faiss
import tensorflow as tf

# USE 모델 로드 (처음 실행 시 다운로드될 수 있습니다.)        
mydir = os.path.dirname(__file__)
embed = hub.load(os.path.join(mydir, "models", "universal-sentence-encoder-tensorflow2-large-v2")) #모델 로드
print("USE model loaded")

# JSON 파일 경로
CHAT_HISTORY_FILE = "chat_history_embeddings.json"

def load_chat_history():
    """채팅 기록 및 임베딩을 로드합니다."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data.get('history', []), data.get('embeddings', [])
            except json.JSONDecodeError:
                print("경고: 채팅 기록 파일이 손상되었습니다. 초기화합니다.")
                return [], []
    return [], []

def save_chat_history(history, embeddings):
    """채팅 기록 및 임베딩을 저장합니다."""
    data = {"history": history, "embeddings": embeddings}
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def vectorize_text(text):
    """텍스트를 USE 임베딩 벡터로 변환합니다."""
    return embed([text])[0].numpy().tolist()

def add_chat_log(speaker, message, history, embeddings):
    """새로운 채팅 기록을 추가하고 벡터화하여 저장합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append({"timestamp": timestamp, "speaker": speaker, "text": message})
    embeddings.append(vectorize_text(message))
    save_chat_history(history, embeddings)
    #print(f"[{timestamp}] {speaker}: {message}")
    return history, embeddings # 업데이트된 history와 embeddings 반환

def attention_weighted_average(query_embedding, embeddings):
    """주의 메커니즘을 사용하여 임베딩을 가중 평균합니다."""
    query_tensor = tf.constant([query_embedding], dtype=tf.float32)
    embeddings_tensor = tf.constant(embeddings, dtype=tf.float32)

    attention_scores = tf.matmul(query_tensor, embeddings_tensor, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores)
    weighted_average = tf.matmul(attention_weights, embeddings_tensor)
    return weighted_average.numpy().tolist()[0]

def find_relevant_memory(query, history, embeddings, top_n=3):
    """현재 쿼리와 의미적으로 가장 유사한 과거 대화 내용을 찾습니다."""
    if not embeddings:
        return []

    query_embedding = vectorize_text(query)
    weighted_embedding = attention_weighted_average(query_embedding, embeddings)

    # Faiss를 사용하여 유사도 검색
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    distances, indices = index.search(np.array([weighted_embedding]).astype('float32'), top_n)

    relevant_memories = []
    for i in range(min(top_n, len(indices[0]))):
        relevant_memories.append(history[indices[0][i]])

    return relevant_memories

