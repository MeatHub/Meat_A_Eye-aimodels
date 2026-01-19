import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob

# 1. 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\meat_vision_v2.pth"
TEST_IMAGE_DIR = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\test_data" # 테스트 이미지가 있는 폴더
CLASS_NAMES = ['Beef chuck', 'Beef fillet', 'Beef flank', 'Beef round', 'Liver', 'Roast beef', 'Strip-lion']

# 2. 모델 로드 함수
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# 3. 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict():
    model = load_model()
    image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.*"))
    
    if not image_files:
        print(f"에러: {TEST_IMAGE_DIR} 폴더에 이미지가 없습니다.")
        return

    print(f"\n{'파일명':<20} | {'예측 결과':<15} | {'신뢰도':<10}")
    print("-" * 50)

    with torch.no_grad():
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                filename = os.path.basename(img_path)
                label = CLASS_NAMES[pred.item()]
                confidence = conf.item() * 100

                print(f"{filename:<20} | {label:<15} | {confidence:>6.2f}%")
            except Exception as e:
                print(f"{os.path.basename(img_path)} 처리 중 에러 발생: {e}")

if __name__ == "__main__":
    predict()