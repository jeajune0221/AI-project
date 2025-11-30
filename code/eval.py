import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from model import HandGestureCNN, HandDataset, get_device  # model.py 에 있는 거 재사용

def main():
    device = get_device()

    # 1) 전처리 (학습 때 쓴 거 그대로)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])

    # 2) 전체 데이터셋 로드
    dataset = HandDataset(root_dir="dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3) 모델/가중치 로드
    model = HandGestureCNN(num_classes=3).to(device)
    state = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    total = 0
    correct = 0

    # 클래스별 정확도 계산
    num_classes = 3
    class_total = [0] * num_classes
    class_correct = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)          
            preds = outputs.argmax(dim=1)       

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for c in range(num_classes):
                mask = (labels == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == labels) & mask).sum().item()

    print(f"전체 정확도: {correct/total*100:.2f}%")

    class_names = ["LEFT", "RIGHT", "OTHER"]
    for c in range(num_classes):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c] * 100
            print(f"{class_names[c]} 정확도: {acc:.2f}% ( {class_correct[c]} / {class_total[c]} )")

if __name__ == "__main__":
    main()
