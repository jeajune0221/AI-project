import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


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
        self.label_map = {
            "left": 0,
            "right": 1,
            "other": 2,
        }

        # 이미지 경로와 라벨을 저장할 리스트
        self.image_paths = []
        self.labels = []

        # 지원할 확장자
        valid_exts = (".png", ".jpg", ".jpeg")

        # 각 클래스 폴더 순회
        for cls_name, label in self.label_map.items():
            folder = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(folder):
                # 폴더가 없으면 건너뜀
                print(f"[경고] 폴더 없음: {folder}")
                continue

            # 파일 이름들을 정렬된 순서로 읽기
            file_list = sorted(os.listdir(folder))
            for fname in file_list:
                if not fname.lower().endswith(valid_exts):
                    continue
                full_path = os.path.join(folder, fname)
                self.image_paths.append(full_path)
                self.labels.append(label)

        if len(self.image_paths) == 0:
            print("[주의] 데이터셋에 이미지가 하나도 없습니다.")

        print(f"[HandDataset] 총 이미지 개수: {len(self.image_paths)}")

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

        # 혹시 사이즈가 다를 수 있으니 224x224로 맞춰줌
        img = cv2.resize(img, (224, 224))

        # transform 적용 (ToTensor + Normalize 등)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # transform이 없으면 직접 Tensor로 변환 (비추천, 그냥 참고용)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # label은 int로 두면 DataLoader가 LongTensor로 묶어줌
        return img, label


# 2. CNN용 전처리(transform)
# ToTensor: (H,W,C) [0~255] -> (C,H,W) [0~1]
# Normalize: 대략 -1~1 근처 값으로 스케일링
train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])



# 3. Dataset / DataLoader 생성
# root_dir는 left/right/other 폴더들의 상위 폴더
train_dataset = HandDataset(
    root_dir="dataset",
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,   # 배치 크기, 나중에 바꿔도 됨
    shuffle=True,
    num_workers=0,   # 처음엔 0으로 두고 디버깅
)



# 4. 테스트
if __name__ == "__main__":
    if len(train_dataset) == 0:
        print("데이터가 없어서 배치를 뽑을 수 없습니다.")
    else:
        images, labels = next(iter(train_loader))
        print("images.shape:", images.shape)   # 예: torch.Size([32, 3, 224, 224])
        print("labels.shape:", labels.shape)   # 예: torch.Size([32])
        print("labels[:10]:", labels[:10])
