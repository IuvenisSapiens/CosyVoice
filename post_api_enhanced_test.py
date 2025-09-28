import requests  # 导入requests库，用于发送HTTP请求
import json  # 导入json库，用于处理JSON数据

headers = {"Content-Type": "application/json"}  # 设置请求头，指定内容类型为JSON

gpt = {
    "text": """n阶方阵A可对角化的充要条件是它具有n个线性无关的特征向量. 属于不同特征值的特征向量是线性无关的. 若n阶方阵A有n个互异的特征值, 则A必可对角化. 不能对角化的矩阵一定具有多重特征值, 对于不能对角化的矩阵也希望找到某种标准形式, 使之尽量接近对角化的形式, 这就是我们要介绍的矩阵的若当标准形.""",  # 要合成的文本
    "new": 1,  # 是否使用自定义音色，1表示使用，0表示不使用
    "speaker": "团长_愤怒_v2",  # 指定语音合成的说话人
    "streaming": 0,  # 是否使用流式合成，1表示使用，0表示不使用
    "speed": 1,  # 语速，范围0.5-2.0
    "volume": 1,  # 音量，范围0.5-2.0
}

response = requests.post(
    "http://localhost:9880/",  # API的URL
    data=json.dumps(gpt),  # 将gpt字典转换为JSON字符串作为请求体
    headers=headers,  # 设置请求头
)

audio_data = response.content  # 获取响应的二进制内容，即音频数据

with open("post请求测试.wav", "wb") as f:  # 以二进制写模式打开文件
    f.write(audio_data)  # 将音频数据写入文件
