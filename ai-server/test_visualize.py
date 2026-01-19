import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob

# 1. 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\meat_vision_v2.pth"
TEST_IMAGE_DIR = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\test_data"
RESULT_DIR = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\test_results"
CLASS_NAMES = ['Beef chuck', 'Beef fillet', 'Beef flank', 'Beef round', 'Liver', 'Roast beef', 'Strip-lion']

os.makedirs(RESULT_DIR, exist_ok=True)

# 2. Grad-CAM 클래스 정의
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        target_layer.register_forward_hook(self.save_gradient)
        target_layer.register_full_backward_hook(self.save_gradient_backward)

    def save_gradient(self, module, input, output):
        self.output = output

    def save_gradient_backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.output, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

# 3. 모델 로드 및 설정
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# EfficientNet-B0의 마지막 컨볼루션 레이어 선택
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_visual_test():
    image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.*"))
    
    for img_path in image_files:
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(raw_img, (224, 224))
        
        input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)
        
        # 추론
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
        class_idx = pred.item()
        
        # 히트맵 생성
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 원본 이미지와 히트맵 합성
        result_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        
        # 결과 저장
        filename = os.path.basename(img_path)
        save_path = os.path.join(RESULT_DIR, f"res_{filename}")
        
        # 텍스트 추가 (예측값, 신뢰도)
        text = f"{CLASS_NAMES[class_idx]} ({conf.item()*100:.1f}%)"
        cv2.putText(result_img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"저장 완료: {save_path}")

if __name__ == "__main__":
    run_visual_test()