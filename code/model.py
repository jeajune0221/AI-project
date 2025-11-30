# [중요] 전체 구조
# 1) Seed/Device 설정
# 2) HandDataset (left/right/other 이미지 로더)
# 3) HandGestureCNN (순전파 모델 정의)
# 4) train/eval (순전파 + 역전파 + 정확도 계산)
# 5) main() (하이퍼파라미터 설정 + 학습 루프 + 모델 저장)

import os
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as T


# =========================================================
# 1. Seed / Device 설정
# =========================================================

def set_seed(seed: int = 42):
    """실험 재현성을 위한 랜덤 시드 고정"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """CPU / CUDA / MPS 중 사용 가능한 디바이스 선택"""
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[Device] Using Apple MPS")
        return torch.device("mps")
    else:
        print("[Device] Using CPU")
        return torch.device("cpu")


# =========================================================
# 2. HandDataset 정의 (dataset/left, right, other 읽기)
# =========================================================

class HandDataset(Dataset):
    """
    dataset/left, dataset/right, dataset/other 구조에서
    이미지를 읽어와 (image_tensor, label_int)로 넘겨주는 Dataset
    """
    def __init__(self, root_dir: str, transform=None):
        """
        root_dir: 예) "dataset"
                  내부에 left/, right/, other/ 폴더가 있다고 가정
        transform: torchvision.transforms.Compose 같은 전처리 함수
        """
        self.root_dir = root_dir
        self.transform = transform

        # 라벨 이름 -> 숫자 매핑
        # 0: left, 1: right, 2: other  (방향 제스처 기준으로 촬영했다고 가정)
        self.label_map = {
            "left": 0,
            "right": 1,
            "other": 2,
        }

        self.image_paths = []
        self.labels = []

        valid_exts = (".png", ".jpg", ".jpeg")

        # 각 클래스 폴더에서 이미지 경로 수집
        for cls_name, label in self.label_map.items():
            folder = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(folder):
                print(f"[경고] 폴더 없음: {folder}")
                continue

            file_list = sorted(os.listdir(folder))
            for fname in file_list:
                if not fname.lower().endswith(valid_exts):
                    continue
                full_path = os.path.join(folder, fname)
                self.image_paths.append(full_path)
                self.labels.append(label)

        if len(self.image_paths) == 0:
            print("[주의] 데이터셋에 이미지가 하나도 없습니다.")

        # ---- 클래스별 개수 집계 (왼손 부족 여부 확인용) ----
        self.num_classes = 3
        self.class_counts = [0] * self.num_classes
        for lb in self.labels:
            self.class_counts[lb] += 1

        print(f"[HandDataset] 총 이미지 개수: {len(self.image_paths)}")
        print(f"  - left (0)  : {self.class_counts[0]}")
        print(f"  - right (1) : {self.class_counts[1]}")
        print(f"  - other (2) : {self.class_counts[2]}")

    def __len__(self):
        # 전체 샘플 개수
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        idx번째 (이미지, 라벨) 하나를 읽어서 반환
        반환 형식: (image_tensor, label_int)
        - image_tensor: (3, 224, 224) 형태의 torch.Tensor
        - label_int: 0/1/2 중 하나 (left/right/other)
        """
        path = self.image_paths[idx]
        label = self.labels[idx]

        # 이미지 로드 (BGR)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"이미지 로드 실패: {path}")

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 혹시 사이즈가 다를 수 있으니 224x224로 맞춤
        img = cv2.resize(img, (224, 224))

        # transform 적용 (ToTensor + Normalize)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # transform 미사용 시: 기본 텐서 변환 (참고용)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


# =========================================================
# 3. CNN 모델 정의 (HandGestureCNN)
#    입력: (N, 3, 224, 224) → 출력: (N, 3)
# =========================================================

class HandGestureCNN(nn.Module):
    """손 제스처 분류용 간단한 CNN (left/right/other, 3클래스)"""
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # 특징 추출부 (Conv + BatchNorm + ReLU + MaxPool)
        self.features = nn.Sequential(
            # Block 1: 3x224x224 -> 16x112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x112x112 -> 32x56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32x56x56 -> 64x28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv 종료 후 feature map 크기: 64 x 28 x 28
        conv_out_dim = 64 * 28 * 28  # = 50176

        # 분류기 (FC + ReLU + Dropout + FC)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (N, 3, 224, 224)
        x = self.features(x)            # -> (N, 64, 28, 28)
        x = x.view(x.size(0), -1)       # -> (N, 64*28*28)
        x = self.classifier(x)          # -> (N, 3) (로짓)
        return x


# =========================================================
# 4. 한 epoch 학습 / 검증 함수
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """train_loader 한 바퀴: 순전파 + 역전파 + 파라미터 업데이트"""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)               # 순전파
        loss = criterion(outputs, labels)     # 손실 계산

        loss.backward()                       # 역전파
        optimizer.step()                      # 가중치 갱신

        # 통계 (배치 단위 누적)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    """val_loader 한 바퀴: 평가 전용 (역전파 없음)"""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc



# 5. 메인 학습 루프 (하이퍼파라미터 포함)
def main():
    """
    전체 학습 파이프라인:
    - 데이터 준비 + 클래스 불균형 보정
    - 학습 + 검증
    - 최고 성능 모델 저장
    """
    set_seed(42)
    device = get_device()

    # 5-1. Transform 정의 (CNN 입력 전처리)
    # 주의: left/right를 섞지 않기 위해 HorizontalFlip 같은 건 넣지 않음
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])

    # 5-2. Dataset 생성 + Train/Val 분할
    full_dataset = HandDataset(root_dir="dataset", transform=train_transform)

    if len(full_dataset) == 0:
        print("데이터가 없어서 학습을 진행할 수 없습니다.")
        return

    train_ratio = 0.8
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[Split] train: {len(train_dataset)} / val: {len(val_dataset)}")

    # 5-2-1. 클래스 불균형 보정 (WeightedRandomSampler)
    # full_dataset.labels 를 활용해서 train 부분의 class weight 계산
    num_classes = 3
    class_counts = [0] * num_classes
    for idx in train_dataset.indices:
        lb = full_dataset.labels[idx]
        class_counts[lb] += 1

    print("[Train 클래스 분포]")
    print(f"  left (0)  : {class_counts[0]}")
    print(f"  right (1) : {class_counts[1]}")
    print(f"  other (2) : {class_counts[2]}")

    # count 가 0인 클래스는 없다고 가정 (있으면 데이터 추가 필요)
    class_weights = [0.0] * num_classes
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_weights[c] = 1.0 / class_counts[c]
        else:
            class_weights[c] = 0.0

    # 각 샘플별 weight 리스트
    sample_weights = []
    for idx in train_dataset.indices:
        lb = full_dataset.labels[idx]
        sample_weights.append(class_weights[lb])
    sample_weights = torch.DoubleTensor(sample_weights)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    #5-3. DataLoader 생성
    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # sampler 사용 시 shuffle=False
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 5-4. 모델 / 손실함수 / 옵티마이저 / 스케줄러 / Epoch 수
    model = HandGestureCNN(num_classes=3).to(device)

    # 클래스 weight 를 손실함수에도 반영 (left 샘플이 적어도 가중치 ↑)
    ce_class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_class_weights)

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 30  # 필요하면 조절

    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    # Epoch 반복 학습 루프 
    for epoch in range(1, num_epochs + 1):
        print(f"\n[Epoch {epoch}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        # 최고 검증 정확도 갱신 시, 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated! (Val Acc = {best_val_acc*100:.2f}%)  Saved to {best_model_path}")

    print("\n학습 완료!")
    print(f"최고 검증 정확도: {best_val_acc*100:.2f}%")
    print(f"베스트 모델 파일: {best_model_path}")


if __name__ == "__main__":
    main()
