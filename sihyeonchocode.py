import os
import cv2
import numpy as np

# 1) 데이터 불러오기 & 전처리
data_dir = 'dataset'                   # dataset/apple, dataset/speaer 준비
classes  = ['apple', 'speaer']         # 0: apple, 1: speaer

X, y = [], []
for label, cls in enumerate(classes):
    folder = os.path.join(data_dir, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        X.append(img.flatten())
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# 2) 학습/테스트 데이터 분리 (80% 학습, 20% 테스트)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# 3) 피처 스케일링
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4) 차원 축소 (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=100)    # 4096 → 100차원
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

# 5) 모델 정의
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
models = {
    'SVM':      SVC(kernel='linear', C=0.001),
    'Logistic': LogisticRegression(max_iter=500, C=0.001)
}

# 6) 학습·평가·혼동행렬 출력
from sklearn.metrics import accuracy_score, confusion_matrix
for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)

    # 예측
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    # 정확도 계산
    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)

    # 혼동행렬 계산
    cm_train = confusion_matrix(y_train, train_preds)
    cm_test  = confusion_matrix(y_test,  test_preds)

    # 결과 출력
    print(f"\n=== {name} ===")
    print(f"학습 정확도: {train_acc*100:.1f}%")
    print(f"테스트 정확도: {test_acc*100:.1f}%")
    print("학습 혼동행렬:")
    print(cm_train)
    print("테스트 혼동행렬:")
    print(cm_test)