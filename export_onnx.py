import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random_seed = 777


def seed_everything(num=777):
    torch.manual_seed(num)
    # torch.cuda.manual_seed(num)
    # torch.cuda.manual_seed_all(num)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(num)
    random.seed(num)


seed_everything(random_seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)

        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(5, 10, 3, batch_first=True)

    def forward(self, x):
        # print(x.shape)
        x, (hidden, cell) = self.lstm(x)
        return x, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, lstm_output, final_state):
        hidden = final_state.view(-1, 10, 3)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(1)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(1)

        return context


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return torch.sigmoid(x)


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()

        self.CNN = CNN()
        self.LSTM = LSTM()
        self.Attention = Attention()
        self.Dense = Dense()

    def forward(self, x):
        x = self.CNN(x)

        state, hidden = self.LSTM(x)

        context = self.Attention(state, hidden)
        x = self.Dense(context)
        return x


model = OurModel()
from torchinfo import summary

# summary(model, [32, 1, 20])

import time

start = time.time()  # 시작 시간 저장
from tqdm import tqdm

x = torch.empty(1, 1, 20)

# 모델 변환
torch.onnx.export(
    model,  # 실행될 모델
    x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
    "out_model.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
    export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
    opset_version=10,  # 모델을 변환할 때 사용할 ONNX 버전
    do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
    input_names=["input"],  # 모델의 입력값을 가리키는 이름
    output_names=["output"],
)  # 모델의 출력값을 가리키는 이름)
