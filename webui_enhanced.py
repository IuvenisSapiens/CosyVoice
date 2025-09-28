# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhang Senlin)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import ffmpeg
import shutil
from funasr import AutoModel
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"


model_dir = "iic/SenseVoiceSmall"
asr_model = AutoModel(
    model=model_dir, disable_update=True, log_level="DEBUG", device="cuda:0"
)


def prompt_wav_recognition(prompt_wav):
    res = asr_model.generate(
        input=prompt_wav,
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
    )
    text = res[0]["text"].split("|>")[-1]
    return text


def recognize_audio(uploaded_audio, recorded_audio):
    # 判断用户输入从何而来，并调用相应的识别函数
    if uploaded_audio:
        return prompt_wav_recognition(uploaded_audio)  # 处理上传的音频
    elif recorded_audio:
        return prompt_wav_recognition(recorded_audio)  # 处理录制的音频
    else:
        return "没有音频文件可供识别"


inference_mode_list = ["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"]
instruct_dict = {
    "预训练音色": "1. 选择预训练音色\n2. 点击生成音频按钮",
    "3s极速复刻": "1. 选择（上传）prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，则优先使用选择（上传）的prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮",
    "跨语种复刻": "1. 选择（上传）prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，则优先使用选择（上传）的prompt音频文件\n2. 点击生成音频按钮",
    "自然语言控制": "1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮",
}
stream_mode_list = [("否", False), ("是", True)]
max_val = 0.8


reference_wavs = ["请选择参考音频或者自己上传"]
for name in os.listdir(f"{ROOT_DIR}/参考音频/"):
    reference_wavs.append(name)

spk_new = ["无"]

for name in os.listdir(f"{ROOT_DIR}/voices/"):
    # print(name.replace(".pt",""))
    spk_new.append(name.replace(".pt", ""))


def adjust_volume_speed(input_audio: np.ndarray, speed: float, volume: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")

    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input(
        "pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1
    )

    # 变速处理
    # 音量调节处理
    output_stream = input_stream.filter("atempo", speed).filter("volume", volume=volume)

    # 输出流到管道
    out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
        input=raw_audio, capture_stdout=True, capture_stderr=True
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


def refresh_choices():

    spk_new = ["无"]

    for name in os.listdir(f"{ROOT_DIR}/voices/"):
        # print(name.replace(".pt",""))
        spk_new.append(name.replace(".pt", ""))

    return {"choices": spk_new, "__type__": "update"}


def change_choices():

    reference_wavs = ["选择参考音频,或者自己上传"]

    for name in os.listdir(f"{ROOT_DIR}/参考音频/"):
        reference_wavs.append(name)

    return {"choices": reference_wavs, "__type__": "update"}


def change_wav(audio_path):

    text = audio_path.replace(".wav", "").replace(".mp3", "").replace(".WAV", "")

    return f"{ROOT_DIR}/参考音频/{audio_path}", text


def save_name(name):

    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False

    shutil.copyfile(f"{ROOT_DIR}/output.pt", f"{ROOT_DIR}/voices/{name}.pt")
    gr.Info("音色保存成功,存放位置为voices目录")


def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat(
        [speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1
    )
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(
    tts_text,
    mode_checkbox_group,
    sft_dropdown,
    prompt_text,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text,
    seed,
    stream,
    speed,
    speed_factor,
    volume_factor,
    new_dropdown,
):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ["自然语言控制"]:
        if "CosyVoice2" in args.model_dir:
            if instruct_text == "":
                gr.Warning("您正在使用自然语言控制模式, 请输入instruct文本")
                yield (cosyvoice.sample_rate, default_data)
        else:
            if cosyvoice.instruct is False:
                gr.Warning(
                    "您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型".format(
                        args.model_dir
                    )
                )
                yield (cosyvoice.sample_rate, default_data)
            if instruct_text == "":
                gr.Warning("您正在使用自然语言控制模式, 请输入instruct文本")
                yield (cosyvoice.sample_rate, default_data)
            if prompt_wav is not None or prompt_text != "":
                gr.Info("您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略")
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ["跨语种复刻"]:
        if cosyvoice.instruct is True:
            gr.Warning(
                "您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型".format(
                    args.model_dir
                )
            )
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != "":
            gr.Info("您正在使用跨语种复刻模式, instruct文本会被忽略")
        if prompt_wav is None:
            gr.Warning("您正在使用跨语种复刻模式, 请提供prompt音频")
            yield (cosyvoice.sample_rate, default_data)
        gr.Info("您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言")
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ["3s极速复刻", "跨语种复刻"]:
        if prompt_wav is None:
            gr.Warning("prompt音频为空，您是否忘记输入prompt音频？")
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(
                "prompt音频采样率{}低于{}".format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr
                )
            )
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ["预训练音色"]:
        if instruct_text != "" or prompt_wav is not None or prompt_text != "":
            gr.Info(
                "您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！"
            )
        if sft_dropdown == "":
            gr.Warning("没有可用的预训练音色！")
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ["3s极速复刻"]:
        if prompt_text == "":
            gr.Warning("prompt文本为空，您是否忘记输入prompt文本？")
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != "":
            gr.Info("您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！")

    if mode_checkbox_group == "预训练音色":
        logging.info("get sft inference request")
        set_all_random_seed(seed)
        if stream:
            for i in cosyvoice.inference_sft(
                tts_text,
                sft_dropdown,
                stream=stream,
                speed=speed,
                new_dropdown=new_dropdown,
            ):
                if speed_factor != 1.0 or volume_factor != 1.0:
                    try:
                        numpy_array = i["tts_speech"].numpy()
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(
                            audio,
                            speed=speed_factor,
                            volume=volume_factor,
                            sr=int(22050),
                        )
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = i["tts_speech"].numpy().flatten()
                yield (cosyvoice.sample_rate, audio_data)
        else:
            tts_speeches = []
            for i in cosyvoice.inference_sft(
                tts_text,
                sft_dropdown,
                stream=stream,
                speed=speed,
                new_dropdown=new_dropdown,
            ):
                tts_speeches.append(i["tts_speech"])
            numpy_array = torch.concat(tts_speeches, dim=1).numpy()
            if speed_factor != 1.0 or volume_factor != 1.0:
                try:
                    audio = (numpy_array * 32768).astype(np.int16)
                    audio_data = adjust_volume_speed(
                        audio,
                        speed=speed_factor,
                        volume=volume_factor,
                        sr=int(22050),
                    )
                except Exception as e:
                    print(f"Failed to process audio: \n{e}")
            else:
                audio_data = numpy_array.flatten()
            yield (cosyvoice.sample_rate, audio_data)

    elif mode_checkbox_group == "3s极速复刻":
        logging.info("get zero_shot inference request")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        if stream:
            for i in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                if speed_factor != 1.0 or volume_factor != 1.0:
                    try:
                        numpy_array = i["tts_speech"].numpy()
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(
                            audio,
                            speed=speed_factor,
                            volume=volume_factor,
                            sr=int(22050),
                        )
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = i["tts_speech"].numpy().flatten()
                yield (cosyvoice.sample_rate, audio_data)
        else:
            tts_speeches = []
            for i in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                tts_speeches.append(i["tts_speech"])
            numpy_array = torch.concat(tts_speeches, dim=1).numpy()
            if speed_factor != 1.0 or volume_factor != 1.0:
                try:
                    audio = (numpy_array * 32768).astype(np.int16)
                    audio_data = adjust_volume_speed(
                        audio,
                        speed=speed_factor,
                        volume=volume_factor,
                        sr=int(22050),
                    )
                except Exception as e:
                    print(f"Failed to process audio: \n{e}")
            else:
                audio_data = numpy_array.flatten()
            yield (cosyvoice.sample_rate, audio_data)
    elif mode_checkbox_group == "跨语种复刻":
        logging.info("get cross_lingual inference request")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        if stream:
            for i in cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                if speed_factor != 1.0 or volume_factor != 1.0:
                    try:
                        numpy_array = i["tts_speech"].numpy()
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(
                            audio,
                            speed=speed_factor,
                            volume=volume_factor,
                            sr=int(22050),
                        )
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = i["tts_speech"].numpy().flatten()
                yield (cosyvoice.sample_rate, audio_data)
        else:
            tts_speeches = []
            for i in cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                tts_speeches.append(i["tts_speech"])
            numpy_array = torch.concat(tts_speeches, dim=1).numpy()
            if speed_factor != 1.0 or volume_factor != 1.0:
                try:
                    audio = (numpy_array * 32768).astype(np.int16)
                    audio_data = adjust_volume_speed(
                        audio,
                        speed=speed_factor,
                        volume=volume_factor,
                        sr=int(22050),
                    )
                except Exception as e:
                    print(f"Failed to process audio: \n{e}")
            else:
                audio_data = numpy_array.flatten()
            yield (cosyvoice.sample_rate, audio_data)
    else:
        logging.info("get instruct inference request")
        set_all_random_seed(seed)
        if "CosyVoice2" in args.model_dir:
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            if stream:
                for i in cosyvoice.inference_instruct2(
                    tts_text,
                    instruct_text,
                    prompt_speech_16k,
                    stream=stream,
                    speed=speed,
                ):
                    if speed_factor != 1.0 or volume_factor != 1.0:
                        try:
                            numpy_array = i["tts_speech"].numpy()
                            audio = (numpy_array * 32768).astype(np.int16)
                            audio_data = adjust_volume_speed(
                                audio,
                                speed=speed_factor,
                                volume=volume_factor,
                                sr=int(22050),
                            )
                        except Exception as e:
                            print(f"Failed to process audio: \n{e}")
                    else:
                        audio_data = i["tts_speech"].numpy().flatten()
                    yield (cosyvoice.sample_rate, audio_data)
            else:
                tts_speeches = []
                for i in cosyvoice.inference_instruct2(
                    tts_text,
                    instruct_text,
                    prompt_speech_16k,
                    stream=stream,
                    speed=speed,
                ):
                    tts_speeches.append(i["tts_speech"])
                numpy_array = torch.concat(tts_speeches, dim=1).numpy()
                if speed_factor != 1.0 or volume_factor != 1.0:
                    try:
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(
                            audio,
                            speed=speed_factor,
                            volume=volume_factor,
                            sr=int(22050),
                        )
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = numpy_array.flatten()
                yield (cosyvoice.sample_rate, audio_data)
        else:
            if stream:
                for i in cosyvoice.inference_instruct(
                    tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed
                ):
                    if speed_factor != 1.0 or volume_factor != 1.0:
                        try:
                            numpy_array = i["tts_speech"].numpy()
                            audio = (numpy_array * 32768).astype(np.int16)
                            audio_data = adjust_volume_speed(
                                audio,
                                speed=speed_factor,
                                volume=volume_factor,
                                sr=int(22050),
                            )
                        except Exception as e:
                            print(f"Failed to process audio: \n{e}")
                    else:
                        audio_data = i["tts_speech"].numpy().flatten()
                    yield (cosyvoice.sample_rate, audio_data)
            else:
                tts_speeches = []
                for i in cosyvoice.inference_instruct(
                    tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed
                ):
                    tts_speeches.append(i["tts_speech"])
                numpy_array = torch.concat(tts_speeches, dim=1).numpy()
                if speed_factor != 1.0 or volume_factor != 1.0:
                    try:
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(
                            audio,
                            speed=speed_factor,
                            volume=volume_factor,
                            sr=int(22050),
                        )
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = numpy_array.flatten()
                yield (cosyvoice.sample_rate, audio_data)


def main():
    with gr.Blocks() as demo:
        gr.Markdown(
            "### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)"
        )
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")
        gr.Markdown(
            "#### `<|im_start|>`, `<|im_end|>`, `<|endofprompt|>`, `[breath]`, `<strong>`, `</strong>`, `[noise]`, `[laughter]`, `[cough]`, `[clucking]`, `[accent]`, `[quick_breath]`, `<laughter>`, `</laughter>`, `[hissing]`, `[sigh]`, `[vocalized-noise]`, `[lipsmack]`, `[mn]`"
        )

        example_tts_text = [
            "我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
            "我们走的每一步，都是我们策略的一部分；你看到的所有一切，包括我此刻与你交谈，所做的一切，所说的每一句话，都有深远的含义。",
            "那位喜剧演员真有才，[laughter]一开口就让全场观众爆笑。",
            "他搞的一个恶作剧，让大家<laughter>忍俊不禁</laughter>。",
            "希望你以后能够做的比我还好呦。",
            "I am a newly launched generative speech large model by the Qwen Voice Team of the Tongyi Laboratory, offering comfortable and natural text-to-speech synthesis capabilities.",
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "用粤语说这句话<|endofprompt|>我最近迷上一部经典港剧，入面嗰啲对白真系有嚟头。",
        ]
        tts_text = gr.Textbox(
            label="输入合成文本",
            lines=1,
            value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
        )
        gr.Examples(label="示例文本", examples=example_tts_text, inputs=[tts_text])
        gr.Examples(
            label="笑声控制",
            examples=[
                "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
                "他搞的一个恶作剧，让大家<laughter>忍俊不禁</laughter>。",
                "Oh wow [laughter], I thought I had seen it all until now [laughter]. Your ability to surprise never ceases to amaze me [laughter].",
            ],
            inputs=[tts_text],
        )
        gr.Examples(
            label="强调控制",
            examples=[
                "追求卓越不是终点，它需要你每天都<strong>付出</strong>和<strong>精进</strong>，最终才能达到巅峰。",
                "With <strong>determination</strong> and <strong>focus</strong>, we can overcome <strong>any challenge</strong>.",
            ],
            inputs=[tts_text],
        )
        gr.Examples(
            label="呼吸控制",
            examples=[
                "当你用心去倾听一首音乐时[breath]，你会开始注意到那些细微的音符变化[breath]，并通过它们感受到音乐背后的情感。",
                "深呼吸[breath]让我们保持冷静[breath]仔细思考这个问题。",
            ],
            inputs=[tts_text],
        )
        gr.Examples(
            label="混合控制",
            examples=[
                "这个笑话太有趣了[laughter]，让我喘口气[breath]，<strong>实在是太好笑了</strong>！",
                "The performance was <strong>breathtaking</strong> [breath], and the audience burst into [laughter] thunderous applause.",
            ],
            inputs=[tts_text],
        )
        speed_factor = gr.Slider(
            minimum=0.25,
            maximum=4,
            step=0.05,
            label="语速调节",
            value=1.0,
            interactive=True,
        )
        volume_factor = gr.Slider(
            minimum=0.25,
            maximum=4,
            step=0.05,
            label="音量调节",
            value=1.0,
            interactive=True,
        )
        with gr.Row():
            with gr.Column(scale=0.15):
                mode_checkbox_group = gr.Radio(
                    choices=inference_mode_list,
                    label="选择推理模式",
                    value=inference_mode_list[0],
                )
                sft_dropdown = gr.Dropdown(
                    choices=sft_spk,
                    label="选择预训练音色",
                    value=sft_spk[0],
                    scale=0.25,
                )
            instruction_text = gr.Text(
                label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=1
            )
            with gr.Column(scale=0.25):
                refresh_new_button = gr.Button("刷新新增音色")
                new_dropdown = gr.Dropdown(
                    choices=spk_new,
                    label="选择新增音色",
                    value=spk_new[0],
                    interactive=True,
                )
                refresh_new_button.click(
                    fn=refresh_choices, inputs=[], outputs=[new_dropdown]
                )
            with gr.Column(scale=0.25):
                stream = gr.Radio(
                    choices=stream_mode_list,
                    label="是否流式推理",
                    value=stream_mode_list[0][1],
                )
                speed = gr.Number(
                    value=1,
                    label="速度调节(仅支持非流式推理)",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                )
        with gr.Row():
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")
            with gr.Column(scale=0.25):
                refresh_button = gr.Button("刷新参考音频")
                wavs_dropdown = gr.Dropdown(
                    label="参考音频列表",
                    choices=reference_wavs,
                    value="请选择参考音频或者自己上传",
                    interactive=True,
                )
            refresh_button.click(fn=change_choices, inputs=[], outputs=[wavs_dropdown])
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="选择prompt音频文件，注意采样率不低于16khz",
            )
            prompt_wav_record = gr.Audio(
                sources="microphone",
                type="filepath",
                label="录制prompt音频文件",
            )
        prompt_text = gr.Textbox(
            label="输入prompt文本",
            lines=1,
            placeholder="请输入prompt文本，支持自动语音识别，您可以自行修正识别结果...",
            value="",
        )
        recognize_button = gr.Button("识别prompt音频内容")
        recognize_button.click(
            fn=recognize_audio,
            inputs=[prompt_wav_upload, prompt_wav_record],
            outputs=[prompt_text],
        )
        # prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        # prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])
        instruct_text = gr.Textbox(
            label="输入instruct文本",
            lines=1,
            placeholder="请输入instruct文本。例如：用四川话说这句话。",
            value="",
        )
        gr.Examples(
            label="角色扮演控制",
            examples=[
                "神秘",
                "凶猛",
                "好奇",
                "优雅",
                "孤独",
                "模仿机器人风格",
                "我想听听你模仿小猪佩奇的语气",
                "一个活泼、爱冒险的小精灵",
                "一位权威、威严的古代将军",
                "一个忧郁的诗人",
            ],
            inputs=[instruct_text],
        )
        gr.Examples(
            label="方言控制",
            examples=[
                "用四川话说这句话",
                "用粤语说这句话",
                "用上海话说这句话",
                "用郑州话说这句话",
                "用长沙话说这句话",
                "用天津话说这句话",
            ],
            inputs=[instruct_text],
        )
        gr.Examples(
            label="情感风格",
            examples=[
                "用开心的语气说",
                "用伤心的语气说",
                "用惊讶的语气说",
                "用生气的语气说",
                "用恐惧的情感表达",
                "用恶心的情感表达",
            ],
            inputs=[instruct_text],
        )
        gr.Examples(
            label="语速控制",
            examples=[
                "快速",
                "非常快速",
                "慢速",
                "非常慢速",
            ],
            inputs=[instruct_text],
        )
        gr.Examples(
            label="语气控制",
            examples=[
                "冷静",
                "严肃",
            ],
            inputs=[instruct_text],
        )
        gr.Examples(
            label="英文风格",
            examples=[
                "Selene 'Moonshade', is a mysterious, elegant dancer with a connection to the night. Her movements are both mesmerizing and deadly. ",
                "A female speaker with normal pitch, slow speaking rate, and sad emotion.",
                "Bubbling with happiness",
                "Overcome with sorrow",
                "Speaking very fast",
                "Speaking with patience",
            ],
            inputs=[instruct_text],
        )
        new_name = gr.Textbox(
            label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value=""
        )

        save_button = gr.Button("保存刚刚推理的zero-shot音色")

        save_button.click(save_name, inputs=[new_name])

        wavs_dropdown.change(
            change_wav, [wavs_dropdown], [prompt_wav_upload, prompt_text]
        )

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[
                tts_text,
                mode_checkbox_group,
                sft_dropdown,
                prompt_text,
                prompt_wav_upload,
                prompt_wav_record,
                instruct_text,
                seed,
                stream,
                speed,
                speed_factor,
                volume_factor,
                new_dropdown,
            ],
            outputs=[audio_output],
        )
        mode_checkbox_group.change(
            fn=change_instruction,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text],
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        # default="pretrained_models/CosyVoice-300M",
        # default="pretrained_models/CosyVoice-300M-25Hz",
        # default="pretrained_models/CosyVoice-300M-Instruct",
        # default="pretrained_models/CosyVoice-300M-SFT",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError("no valid model_type!")
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
