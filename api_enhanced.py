import io
import os
import sys
import numpy as np
from flask import Flask, request, Response, send_from_directory
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
import ffmpeg
from flask_cors import CORS
from flask import make_response
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/AcademiCodec".format(ROOT_DIR))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))
cosyvoice = CosyVoice2("pretrained_models/CosyVoice2-0.5B")
# cosyvoice = CosyVoice("pretrained_models/CosyVoice-300M-25Hz")

default_voices = ["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"]

spk_new = []

for name in os.listdir(f"{ROOT_DIR}/voices/"):
    print(name.replace(".pt", ""))
    spk_new.append(name.replace(".pt", ""))

print("默认音色", cosyvoice.list_available_spks())
print("自定义音色", spk_new)

app = Flask(__name__)

CORS(app, cors_allowed_origins="*")

CORS(app, supports_credentials=True)


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


@app.route("/", methods=["POST"])
def sft_post():
    question_data = request.get_json()
    new = question_data.get("new", 0)
    text = question_data.get("text")
    speaker = question_data.get("speaker")
    streaming = question_data.get("streaming", 0)

    speed = question_data.get("speed", 1.0)  # 从 question_data 中获取 speed，默认值为 1.0
    # speed = request.args.get("speed", 1.0) # 从 url 中获取 speed，默认值为 1.0

    volume = question_data.get("volume", 1.0)  # 从 question_data 中获取 volume，默认值为 1.0
    # volume = request.args.get("volume", 1.0)  # 从 url 中获取 volume，默认值为 1.0

    speed = float(speed) # 将 speed 转换为浮点数
    volume = float(volume) # 将 volume 转换为浮点数

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        buffer = io.BytesIO()

        tts_speeches = []

        for i, j in enumerate(
            cosyvoice.inference_sft(
                text, speaker, stream=False, speed=1.0, new_dropdown="无" if new == 0 else speaker
            )
        ):
            # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
            tts_speeches.append(j["tts_speech"])
            
        output = torch.concat(tts_speeches, dim=1)  
            
        if speed != 1.0 or volume != 1.0:
            try:
                numpy_array = output.numpy()
                audio = (numpy_array * 32768).astype(np.int16)
                audio_data = adjust_volume_speed(audio, speed=speed, volume=volume, sr=int(22050))
                audio_data = torch.from_numpy(audio_data)
                audio_data = audio_data.reshape(1, -1)
            except Exception as e:
                print(f"Failed to process audio: \n{e}")
        else:
            audio_data = output
        
        torchaudio.save(buffer, audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        def generate():

            for i, j in enumerate(
                cosyvoice.inference_sft(
                    text, speaker, stream=True, speed=1.0, new_dropdown="无" if new == 0 else speaker
                )
            ):

                tts_speeches = []
                buffer = io.BytesIO()
                tts_speeches.append(j["tts_speech"])
                output = torch.concat(tts_speeches, dim=1)
                
                if speed != 1.0 or volume != 1.0:
                    try:
                        numpy_array = output.numpy()
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(audio, speed=speed, volume=volume, sr=int(22050))
                        audio_data = torch.from_numpy(audio_data)
                        audio_data = audio_data.reshape(1, -1)
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = output
                
                torchaudio.save(buffer, audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers["Content-Type"] = "audio/ogg"
        response.headers["Content-Disposition"] = "attachment; filename=sound.ogg"
        return response


@app.route("/", methods=["GET"])
def sft_get():

    text = request.args.get("text")
    speaker = request.args.get("speaker")
    new = request.args.get("new", 0)
    streaming = request.args.get("streaming", 0)
    speed = request.args.get("speed", 1.0)
    volume = request.args.get("volume", 1.0)
    speed = float(speed)
    volume = float(volume)

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        buffer = io.BytesIO()

        tts_speeches = []

        for i, j in enumerate(
            cosyvoice.inference_sft(
                text, speaker, stream=False, speed=1.0, new_dropdown="无" if new == 0 else speaker
            )
        ):
            # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
            tts_speeches.append(j["tts_speech"])

        output = torch.concat(tts_speeches, dim=1)
        if speed != 1.0 or volume != 1.0:
            try:
                numpy_array = output.numpy()
                audio = (numpy_array * 32768).astype(np.int16)
                audio_data = adjust_volume_speed(audio, speed=speed, volume=volume, sr=int(22050))
                audio_data = torch.from_numpy(audio_data)
                audio_data = audio_data.reshape(1, -1)
            except Exception as e:
                print(f"Failed to process audio: \n{e}")
        else:
            audio_data = output
        torchaudio.save(buffer, audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        def generate():

            for i, j in enumerate(
                cosyvoice.inference_sft(
                    text, speaker, stream=True, speed=1.0, new_dropdown="无" if new == 0 else speaker
                )
            ):

                tts_speeches = []
                buffer = io.BytesIO()
                tts_speeches.append(j["tts_speech"])
                output = torch.concat(tts_speeches, dim=1)
                
                if speed != 1.0 or volume != 1.0:
                    try:
                        numpy_array = output.numpy()
                        audio = (numpy_array * 32768).astype(np.int16)
                        audio_data = adjust_volume_speed(audio, speed=speed, volume=volume, sr=int(22050))
                        audio_data = torch.from_numpy(audio_data)
                        audio_data = audio_data.reshape(1, -1)
                    except Exception as e:
                        print(f"Failed to process audio: \n{e}")
                else:
                    audio_data = output
                
                
                torchaudio.save(buffer, audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers["Content-Type"] = "audio/ogg"
        response.headers["Content-Disposition"] = "attachment; filename=sound.ogg"
        return response

        # return Response(generate(), mimetype='audio/x-wav')


@app.route("/tts_to_audio/", methods=["POST"])
def tts_to_audio():

    import speaker_config

    question_data = request.get_json()
    new = question_data.get("new", 0)
    text = question_data.get("text")
    speaker = speaker_config.speaker
    speed = speaker_config.speed
    volume = speaker_config.volume

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    buffer = io.BytesIO()

    tts_speeches = []

    for i, j in enumerate(
        cosyvoice.inference_sft(
            text, speaker, stream=False, speed=1.0, new_dropdown="无" if new == 0 else speaker
        )
    ):
        # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
        tts_speeches.append(j["tts_speech"])

    output = torch.concat(tts_speeches, dim=1)
    
    
    if speed != 1.0 or volume != 1.0:
        try:
            numpy_array = output.numpy()
            audio = (numpy_array * 32768).astype(np.int16)
            audio_data = adjust_volume_speed(audio, speed=speed, volume=volume, sr=int(22050))
            audio_data = torch.from_numpy(audio_data)
            audio_data = audio_data.reshape(1, -1)
        except Exception as e:
            print(f"Failed to process audio: \n{e}")
    else:
        audio_data = output
    
    torchaudio.save(buffer, audio_data, 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")


@app.route("/speakers", methods=["GET"])
def speakers():

    voices = []

    for x in default_voices:
        voices.append({"name": x, "voice_id": x})

    for name in os.listdir("voices"):
        name = name.replace(".pt", "")
        voices.append({"name": name, "voice_id": name})

    response = app.response_class(
        response=json.dumps(voices), status=200, mimetype="application/json"
    )
    return response


@app.route("/speakers_list", methods=["GET"])
def speakers_list():

    response = app.response_class(
        response=json.dumps(["female_calm", "female", "male"]),
        status=200,
        mimetype="application/json",
    )
    return response


@app.route("/file/<filename>")
def uploaded_file(filename):
    return send_from_directory("音频输出", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9880)
