import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Resnet
from torchvision.models import resnet18
import pytorch_lightning as pl

feature = resnet18(pretrained=True)

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer



# 学習済みモデルのロード
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Streamlitアプリの設定
st.title("犬猫判定アプリ")

st.sidebar.success("Select a demo above.")


image = None
# 画像のアップロード
uploaded_file = st.file_uploader("犬または猫の画像をアップロードしてください(JPG)", type=["jpg", "jpeg"])

picture = st.camera_input("Take a picture")

if picture:
    image = Image.open(picture)
    # st.image(image)
elif uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

if image is not None :
    # 画像の前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0)

    # 画像の予測
    with torch.no_grad():
        predictions = model(img)
        # Softmaxを適用して確率に変換
        # 
        probabilities = F.softmax(predictions, dim=1)
        top_p, top_class = probabilities.topk(1, dim=1)
        predicted_class = top_class[0][0].item()
        predicted_prob = top_p[0][0].item()
        print(predictions)

    # 予測結果の表示
    labels = ['猫', '犬']  # このリストはモデルの訓練時に使ったクラスラベルに合わせてください
    st.write(f"予測結果: {labels[predicted_class]}")
    st.write(f"確信度: {predicted_prob:.2%}")