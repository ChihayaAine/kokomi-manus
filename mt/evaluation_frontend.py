# side by side的对比，用于评测模型的相对能力
import os, sys
import gradio as gr
import random, base64
import time
import numpy as np
import pandas as pd 
import requests
import argparse
from datetime import datetime
import json, yaml
import time
from functools import partial
from threading import Thread
import traceback
import copy, io
import torch
from http import HTTPStatus
from PIL import Image
import csv
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from qwen_agent.tools import TOOL_REGISTRY, BaseTool
from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, FunctionCall, Message

import secrets
import tempfile
import re
from itertools import zip_longest
sys.path.append(os.path.abspath("../../"))
from services.speech import SpeechPlatformTTS, SpeechPlatformASR
from services.s3plus import S3Plus
import fcntl  # Add import for file locking

def get_args():
    parser = argparse.ArgumentParser(f"{__file__}")
    parser.add_argument(f"--cfg_path", type=str, default="../../cfg/evaluation/evaluation_frontend.yaml")
    parser.add_argument(f"--label_save_path", type=str, default="../../output/evaluation/evaluation_frontend_label_result.json")
    parser.add_argument(f"--host", type=str, default="0.0.0.0")
    parser.add_argument(f"--port", type=int, default=8000)
    parser.add_argument(f"--ssl_certfile", type=str, default="../../cfg/ssl_key/cert.pem")
    parser.add_argument(f"--ssl_keyfile", type=str, default="../../cfg/ssl_key/key.pem")

    args = parser.parse_args()
    return args


def read_yaml(path)->dict:
    with open(path, "r") as fp:
        data = yaml.load(fp, Loader=yaml.Loader) 
    return data

# ref: https://www.gradio.app/docs/audio
def generate_tone(mute=False):
    if mute:
        return 16000, np.linspace(0, 1, 1).astype(np.int16)
    
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = random.choice(list(range(len(notes))))
    octave = random.choice([4,5,6])
    duration = 1
    sr = 48000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return sr, audio//8

def append_output_msg(output_msg, append_msg):
    return output_msg+f">{append_msg}\t\t"

def common_clear(t):
    if isinstance(t, str):
        return ""
    elif isinstance(t, list):
        return None
    elif isinstance(t, tuple):
        return None
    else:
        return t


class LLMServer(object):
    def __init__(self, url):
        self.url = url
        self.chat_api = f"{url}/api/chat"
        self.chat_stream_api = f"{url}/api/chat_stream"
        self.default_parameters_api = f"{url}/api/defaut_parameters"
    
    @staticmethod
    def make_input(prompt, history, system):
        messages = []
        if len(system)>0:
            messages.append({"role":"system", "content":system})
        for (query, ans) in history:
            messages.append({"role":"user", "content": query})
            messages.append({"role":"assistant", "content": ans})
        if len(prompt)>0:
            messages.append({"role":"user", "content":prompt})
        return messages
    
    @staticmethod
    def parse_messages(messages:list):
        prompt = ""
        history = []
        system = ""
        if len(messages)>0:
            if messages[0]["role"]=="system":
                system = messages[0]["content"]
                messages.pop(0)
            if len(messages)>0:
                if messages[-1]["role"]=="assistant":
                    prompt = messages[-1]["content"]
                    messages.pop(-1)
            while len(messages)>1:
                q,a = "", ""
                if messages[0]["role"]=="user":
                    q = messages[0]["content"]
                    messages.pop(0)
                if messages[0]["role"]=="assistant":
                    a = messages[0]["content"]
                    messages.pop(0)
                history.append([q,a])
        return prompt, history, system

                                    
    
    def chat(self, messages, params={}):
        json_post_params = {
            "messages": messages,
            "params": params
        }
        resp = f"http_post timeout. api: {self.chat_api}"
        try:
            response = requests.post(url=self.chat_api, json=json_post_params, timeout=30)
            resp = response.json()['data']['text']
        except Exception as e:
            traceback.print_exc()
            resp = str(e)
        finally:
            return resp
        
    def chat_stream(self, messages, params={}):
        json_post_params = {
            "messages": messages,
            "params": params
        }
        print(f"chat_stream. url: {self.chat_stream_api}, param: {json_post_params}")
        return requests.post(url=self.chat_stream_api, json=json_post_params, stream=True, timeout=120)

    def get_default_parameters(self):
        resp = requests.get(self.default_parameters_api, timeout=30)
        return resp.json()["data"]
    

class LLMDebug(object):
    def __init__(self, tts, asr, preset_models):
        self.tts_cfg = copy.deepcopy(tts)
        self.asr_cfg = copy.deepcopy(asr)
        self.preset_models = {m["name"]:m["url"] for m in preset_models}

        self.speech_tts = SpeechPlatformTTS(
            app_key=self.tts_cfg["app_key"], secret_key=self.tts_cfg["secret_key"],
            auth_api=self.tts_cfg["auth_api"], tts_url=self.tts_cfg["tts_url"], tts_api=self.tts_cfg["tts_api"])
        self.speech_asr = SpeechPlatformASR(
            app_key=self.asr_cfg["app_key"], secret_key=self.asr_cfg["secret_key"],
            auth_api=self.asr_cfg["auth_api"], asr_url=self.asr_cfg["asr_url"], asr_api=self.asr_cfg["asr_api"])
        self.tts_voice_names = self.tts_cfg["voice_name"]

    def get_preset_models(self):
        return list(self.preset_models.keys())
    
    def get_tts_voice_names(self):
        return self.tts_voice_names
    
    def select_model_bot(self, model_name, model_url, output_msg):
        model_name_list = gr.Dropdown(choices=list(self.preset_models.keys()))
        if model_name in self.preset_models.keys():
            output_msg = append_output_msg(output_msg, f"url for {model_name} found.")
            return model_name_list, self.preset_models[model_name], output_msg
        else:
            output_msg = append_output_msg(output_msg, f"url for {model_name} not found.")
            return model_name_list, model_url, output_msg
        
    @staticmethod
    def get_default_parameters(url, output_msg):
        default_parameters = {}
        output_msg = "model url is invalid."
        if isinstance(url, str):
            if len(url)>4:
                llm_server = LLMServer(url)
                default_parameters = llm_server.get_default_parameters()
                default_parameters = json.dumps(default_parameters, indent=2)
                output_msg = append_output_msg(output_msg=output_msg, append_msg=f"get_default_parameters successfully.")

        return default_parameters, output_msg


    @staticmethod
    def regenerate_llm(url, system, msg, chatbot, params, output_msg):
        if len(chatbot)>0:
            output_msg = append_output_msg(output_msg, "regenerate...")
            msg = chatbot[-1][0]
            chatbot = chatbot[:-1]
            for resp in LLMDebug.response_wrapper_llm(url, system, msg, chatbot, params, output_msg):
                yield resp
        else:
            msg = "Chatbot has no history..."
            gr.Warning(msg)
            output_msg = append_output_msg(output_msg, msg)
            yield chatbot, output_msg

    @staticmethod
    def render_think(msg:str, start_token='<think>', end_token='</think>'):
        msg = msg.strip()
        if msg.startswith(start_token):
            msg = msg.replace(start_token, '```json')
            if end_token in msg:
                msg = msg.replace(end_token, '```')
            else:
                msg = msg+'\n```'
        return msg

    @staticmethod
    def response_wrapper_llm(url, system, msg, chatbot, params:dict, output_msg):
        llm_server = LLMServer(url=url)
        if len(url.strip())>0:
            start_time = time.time()
            try:
                if isinstance(params, str):
                    if len(params.strip())>0:
                        params = json.loads(params)
                    else:
                        params = {}
                resp = llm_server.chat_stream(LLMServer.make_input(prompt=msg, history=chatbot, system=system), params=params)
                output_msg = append_output_msg(output_msg, "generating...")

                chatbot.append((msg, ""))
                yield chatbot, output_msg
                for line in resp.iter_lines():
                    line = line.decode("utf-8")
                    data_dict = json.loads(line)
                    if not data_dict['finished']:
                        chatbot[-1] = (msg, LLMDebug.render_think(data_dict['data']['text']))
                        yield chatbot, output_msg
                    
                # 流式传输
                text_len = len(chatbot[-1][-1])
                output_msg = append_output_msg(output_msg, 
                    f"{text_len} text/{time.time()-start_time:.2f} s successfully.") 
            except Exception as e:
                output_msg = append_output_msg(output_msg, str(e))
            finally:
                yield chatbot, output_msg
            
        else:
            msg = "bot has no url."
            gr.Warning(msg)
            output_msg = append_output_msg(output_msg, msg)
        yield chatbot, output_msg

    def response_wrapper_tts(self, tts_voice_name, chatbot, output_msg):
        resp_data = {"audio": generate_tone(mute=True), "finished": False}

        def __generate_tts(text, voice_name, resp):
            __rtn_data = self.speech_tts.post(text=text, voice_name=voice_name)
            if __rtn_data["code"]==0:
                resp["audio"] = (__rtn_data['sr'], __rtn_data['data'])
            else:
                print(__rtn_data)
            resp["finished"] = True

        # TTS
        if isinstance(tts_voice_name, str) and tts_voice_name!="off":
            if len(chatbot)>0:
                Thread(target=__generate_tts, kwargs=dict(text=chatbot[-1][-1], voice_name=tts_voice_name, resp=resp_data)).start()
                start_time = time.time()
                time_elaspe = 0
                max_wait_time = 30.0
                output_msg = append_output_msg(output_msg, "generate TTS...")
                while (not resp_data["finished"]) and time_elaspe<max_wait_time:
                    yield resp_data["audio"], output_msg
                    time.sleep(0.5)
                    time_elaspe = time.time()-start_time
                output_msg = append_output_msg(output_msg, f"TTS {time_elaspe:.2f} s successfully.")
            else:
                output_msg = append_output_msg(output_msg, f"no content for data.")
        else:
            output_msg = append_output_msg(output_msg, "TTS OFF.")
        
        yield resp_data["audio"], output_msg
        
    def response_wrapper_asr(self, msg, audio, output_msg):
        # sr, data = audio
        # data = np.flipud(data)
        data_dict = self.speech_asr.post(audio)
        if data_dict['code']==0:
            msg = data_dict["data"]
            output_msg = append_output_msg(output_msg=output_msg, append_msg="ASR successfully.")
        else:
            msg = ""
            output_msg = append_output_msg(output_msg=output_msg, append_msg="ASR failed.")
        return msg, output_msg

#### 部署Qwen-VL-Chat的实现，ref: 
class VLMServer(object):
    def __init__(self, preset_models, s3plus:dict):
        if isinstance(preset_models, list):
            self.preset_models = {m["name"]:m["url"] for m in preset_models}
        else:
            self.preset_models = {}

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            # 对图像进行Base64编码
            base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_encoded_image

    @staticmethod
    def encode_jpeg_image(image_path):
        if isinstance(image_path, str):
            input_image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            input_image = image_path
        else:
            raise TypeError
        # input_image = input_image.resize(size=(self.resized_width, self.resized_height))
        buffer = io.BytesIO()
        input_image.save(buffer, format="jpeg")
        buffer.seek(0)
        # Get the image data as bytes
        image_data = buffer.getvalue()
        # with open(image_path, "rb") as image_file:
            # 对图像进行Base64编码
            # base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded_image
   
    def select_model_bot(self, model_name, model_url, output_msg):
        model_name_list = gr.Dropdown(choices=list(self.preset_models.keys()))
        if model_name in self.preset_models.keys():
            output_msg = append_output_msg(output_msg, f"url for {model_name} found.")
            return model_name_list, self.preset_models[model_name], output_msg
        else:
            output_msg = append_output_msg(output_msg, f"url for {model_name} not found.")
            return model_name_list, model_url, output_msg
    
    def get_preset_models(self):
        return list(self.preset_models.keys())
    
    @staticmethod
    def make_input(prompt, image, history, system):
        messages = []
        if len(system)>0:
            messages.append({"role":"system", "content":system})
        history = history + [[prompt, None]]
        images_temp = []
        if image is not None:
            images_temp = [{"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{VLMServer.encode_jpeg_image(image)}"}}]
        for _, (query, ans) in enumerate(history):
            if isinstance(query, (str, dict)):
                if isinstance(query, str):
                    messages.append({"role":"user", "content": [{"type":"text", "text":query}]+images_temp})
                    images_temp = []
                else: # isinstance(query, dict)[]:
                    images_temp.append({"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{VLMServer.encode_jpeg_image(query['local_path'])}"}})
            else:
                raise TypeError
            
            if ans is not None and len(ans)>0:
                messages.append({"role":"assistant", "content": ans})
        return messages
    
    
    def chat_stream(self, url, image:Image.Image, prompt, history, system, params={}):
        chat_stream_api = f"{url}/api/chat_stream"
        json_post_params = {
            "messages": self.make_input(prompt=prompt, image=image, history=history, system=system),
            "params": params
        }
        print(f"chat_stream. url: {chat_stream_api}, param: {json_post_params}", flush=True)
        print(json_post_params, flush=True)
        resp = requests.post(url=chat_stream_api, json=json_post_params, stream=True)
        for line in resp.iter_lines():
            line = line.decode("utf-8")
            data_dict = json.loads(line)
            if not data_dict['finished']:
                yield data_dict['data']['text']
    
    def predict(self, url, image:Image.Image, query, _chatbot, task_history, system="", params={}):
        resp_iter = self.chat_stream(url=url, image=image, prompt=query, history=task_history, system=system, params=params)
        _chatbot.append([query, None])
        for response in resp_iter:
            print(response)
            _chatbot[-1][-1] = response
            # print(f"predict: {_chatbot}")
            yield _chatbot

        print("VLM: " + _chatbot[-1][-1])
        task_history.append(_chatbot[-1])
        yield _chatbot


    @staticmethod
    def get_default_parameters(url, output_msg):
        default_parameters = {}
        output_msg = "model url is invalid."
        if isinstance(url, str):
            if len(url)>4:
                llm_server = LLMServer(url)
                default_parameters = llm_server.get_default_parameters()
                default_parameters = json.dumps(default_parameters, indent=2)
                output_msg = append_output_msg(output_msg=output_msg, append_msg=f"get_default_parameters successfully.")

        return default_parameters, output_msg


    def regenerate(self, url, image:Image.Image, system, _chatbot, task_history, output_msg):
        if len(url)==0:
            msg = "url is necessary."
            gr.Warning(msg)
            yield _chatbot, task_history, output_msg
        elif not task_history:
            msg = "no chat history."
            gr.Warning(msg)
            yield _chatbot, task_history, output_msg
        else:
            item = task_history[-1]
            if item[1] is None:
                yield _chatbot, task_history, output_msg
            query = item[0]
            _chatbot.pop(-1)
            
            print(f"regenerate: {_chatbot}, {task_history}")
            for _chatbot in self.predict(url, image, query, _chatbot, task_history, system):
                yield _chatbot, task_history, output_msg

    def add_file(self, history, task_history, file, output_msg):
        print(file, flush=True)
        local_path = file
        history = history + [((file,), None)]
        task_history = task_history + [({"local_path": local_path}, None)]
        output_msg = append_output_msg(output_msg, "add file.")
        return history, task_history, output_msg

    def response_wrapper_vlm(self, url, image:Image.Image, query, system, chatbot, task_history, output_msg):
        if len(url)==0 or len(query)==0:
            msg = "url and query are necessary."
            gr.Warning(msg)
            output_msg = append_output_msg(output_msg=output_msg, append_msg=msg)
            yield chatbot, task_history, output_msg
        else:
            # add text
            output_msg = append_output_msg(output_msg=output_msg, append_msg="generating")
            yield chatbot, task_history, output_msg
            for out_chatbot in self.predict(url=url, image=image, query=query, _chatbot=chatbot, task_history=task_history, system=system, params={}):
                yield out_chatbot, task_history, output_msg
            output_msg = append_output_msg(output_msg=output_msg, append_msg="completely.")
            yield out_chatbot, task_history, output_msg
        
#### LLM竞技场
class LLMArena(object):
    def __init__(self, preset_models, question_path, result_path, intro_path):
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.preset_models = copy.deepcopy(preset_models)
        self.question_path = os.path.join(root_folder, question_path)
        self.result_path = os.path.join(root_folder, result_path)
        self.intro_path = os.path.join(root_folder, intro_path)

    def get_question(self):
        questions = pd.read_csv(self.question_path)
        select_idx = random.randint(0, len(questions)-1)
        question = questions.iloc[select_idx]["Prompt"]
        answer = questions.iloc[select_idx]["参考回答"]
        return question, answer
    
    def save_question(self, question, task_type, answer=""):
        data = {"Prompt": [question], "任务类型":[task_type], "参考回答": [answer], "来源": ["LLM-DEV"]}
        pd.DataFrame(data).to_csv(self.question_path, index=False, mode="a", header=False)

    def get_paired_models(self):
        models = copy.deepcopy(self.preset_models)
        random.shuffle(models)
        return models[0], models[1]

    def save_result(self, result, bot1_info, chatbot1, bot2_info, chatbot2):
        bot1_name = bot1_info["name"]
        bot2_name = bot2_info["name"]
        if len(bot1_name)>0 and len(bot2_name)>0 and len(chatbot1)>0 and len(chatbot2)>0 and len(chatbot1[0])==2 and len(chatbot2[0])==2:
            assert chatbot1[0][0]==chatbot2[0][0]
            question = chatbot1[0][0]
            data = {"question":[question], "bot1_name":[bot1_name], "answer1":[chatbot1[0][1]], 
                    "bot2_name":[bot2_name], "answer2":[chatbot2[0][1]], "win":[""], "date":[datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}
            if result=="left_better":
                data["win"][0] = bot1_name
            elif result=="right_better":
                data["win"][0] = bot2_name
            else:
                pass
            if os.path.exists(self.result_path):
                mode, header = "a", False
            else:
                mode, header = "w", True
            pd.DataFrame(data).to_csv(self.result_path, mode=mode, header=header, index=False)
    
    def get_win_rate(self):
        result = pd.read_csv(self.result_path)
        models = copy.deepcopy(self.preset_models)
        for m in models:
            pk_cnt = len([_m for _m in result["bot1_name"].tolist() if _m==m["name"]])+len([_m for _m in result["bot2_name"].tolist() if _m==m["name"]])
            if pk_cnt>5:
                m["win_rate"] = len([_m for _m in result["win"].tolist() if _m==m["name"]])/float(pk_cnt)
            else:
                m["win_rate"] = 0
        models = sorted(models, key=lambda m:m["win_rate"], reverse=True)
        models_df = pd.DataFrame({"Model":[m["name"] for m in models], "Win Rate":[m["win_rate"] for m in models]})
        return models_df
    
    def get_intro(self):
        with open(self.intro_path, mode="r") as fp:
            data = fp.readlines()
        return "".join(data)

#### deployment的调用实现
class LLMDeployment(object):
    def __init__(self, url, preset_models):
        self.url = url
        self.submit_api = f"{url}/api/start_task"
        self.stop_api = f"{url}/api/stop_task"
        self.update_api = f"{url}/api/get_status"
        self.preset_models = copy.deepcopy(preset_models)
        self.dolphin_root = "/mnt/dolphinfs/hdd_pool/docker/user"
        self.model_path_base = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-peisongpa"
        self.model_path_home_base = "/home/hadoop-hmart-peisongpa/dolphinfs_hdd_hadoop-hmart-peisongpa"
        
    def get_preset_models(self):
        return self.preset_models
    
    def __fix_model_path(self, model_path:str):
        if len(model_path)>0:
            if model_path.startswith(self.dolphin_root):
                pass
            elif model_path.startswith(self.model_path_home_base):
                model_path = self.model_path_base+model_path[len(self.model_path_home_base):]
            else:
                if model_path.startswith("/"):
                    model_path = model_path[1:]
                model_path = os.path.join(self.model_path_base, model_path)
        return model_path
                
    def submit_deploy_model(self, user_name, model_type, model_path):
        if len(model_path)==0 or len(model_type)==0 or len(user_name)==0:
            msg = "model_path, model_type and user_name are necessary."
            gr.Warning(msg)
            return {"msg": msg}
        else:
            # model_path = self.__fix_model_path(model_path)
            json_post_params = {
                "user_name": user_name,
                "model_type": model_type,
                "model_path": model_path
            }
            resp = {"msg": f"http_post timeout. api: {self.submit_api}"}
            try:
                response = requests.post(url=self.submit_api, json=json_post_params, timeout=30)
                print(response.json())
                resp = response.json()['data']
            except Exception as e:
                print(f"exception when call {self.submit_api}")
                traceback.print_exc()
                resp = {"msg": str(e)}
            finally:
                return resp
    
    def stop_deploy_model(self, user_name, url, pid):
        if len(url)==0 or len(pid)==0 or len(user_name)==0:
            msg = "url, pid and user_name are necessary for stop."
            gr.Warning(msg)
            yield {"msg": msg}
        else:
            json_post_params = {
                "user_name": user_name,
                "url": url,
                "pid": pid
            }
            yield {"msg": "stop..."}
            resp = {"msg": f"http_post timeout. api: {self.stop_api}"}
            try:
                response = requests.post(url=self.stop_api, json=json_post_params, timeout=30)
                resp = response.json()['data']
            except Exception as e:
                print(f"exception when call {self.stop_api}")
                traceback.print_exc()
                resp = {"msg": str(e)}
            finally:
                yield resp
    
    def update_deploy_status(self):
        resp = {"msg": f"http_post timeout. api: {self.update_api}"}
        try:
            response = requests.get(url=self.update_api, timeout=30)
            resp = response.json()['data']
        except Exception as e:
            traceback.print_exc()
            resp = {"msg": str(e)}
        finally:
            return resp
        
#### Opencompass evaluation的调用实现
class LLMEvalution(object):
    def __init__(self, url, preset_modes, preset_models, preset_datasets):
        self.url = url
        self.submit_api = f"{url}/api/start_task"
        self.stop_api = f"{url}/api/stop_task"
        self.update_api = f"{url}/api/get_status"
        self.preset_modes = copy.deepcopy(preset_modes)
        self.preset_models = copy.deepcopy(preset_models)
        self.preset_datasets = copy.deepcopy(preset_datasets)

    def get_preset_modes(self):
        return self.preset_modes
    
    def get_preset_models(self):
        return self.preset_models
    
    def get_preset_datasets(self, subset=None):
        preset_datasets = []
        for k,v in self.preset_datasets.items():
            if subset is not None:
                if k==subset:
                    preset_datasets += v
        return preset_datasets 
    
    def start_eval_task(self, user_name, mode, model, model_path, datasets1, datasets2):
        if len(model_path)==0 or len(user_name)==0:
            msg = "model_path/url and user_name are necessary."
            gr.Warning(msg)
            yield {"msg":msg}
        elif len(datasets1)+len(datasets2)==0:
            msg = "no dataset is selected."
            gr.Warning(msg)
            yield {"msg":msg}
        else:
            json_post_params = {
                "user_name": user_name,
                "mode": mode,
                "model": model,
                "datasets": datasets1+datasets2,
                "model_path": model_path
            }
            resp = {"msg":"stop..."}
            yield resp
            try:
                response = requests.post(url=self.submit_api, json=json_post_params, timeout=30)
                print(response.json())
                resp = response.json()['data']
            except Exception as e:
                print(f"exception when call {self.submit_api}")
                traceback.print_exc()
                resp = {"msg": str(e)}
            finally:
                yield resp

    def stop_eval_task(self, task_id, user_name, model_path):
        if len(task_id)==0 or len(model_path)==0 or len(user_name)==0:
            msg = "task_id, model_path/url and user_name are necessary for stop."
            gr.Warning(msg)
            yield {"msg": msg}
        else:
            json_post_params = {
                "id": task_id,
                "user_name": user_name,
                "model_path": model_path
            }
            resp = {"msg": "stop..."}
            yield resp
            try:
                response = requests.post(url=self.stop_api, json=json_post_params, timeout=30)
                resp = response.json()['data']
            except Exception as e:
                print(f"exception when call {self.stop_api}")
                traceback.print_exc()
                resp = {"msg": str(e)}
            finally:
                yield resp
        
    def update_eval_status(self):
        resp = {"msg": ""}
        try:
            response = requests.get(url=self.update_api, timeout=30)
            resp = response.json()['data']
        except Exception as e:
            traceback.print_exc()
            resp = {"msg": str(e)}
        finally:
            return resp

class BMLLMEvaluation(object):
    def __init__(self, url, preset_datasets, preset_models):
        self.url = url
        self.submit_api = f"{url}/api/start_task"
        self.stop_api = f"{url}/api/stop_task"
        self.update_api = f"{url}/api/get_status"
        self.preset_datasets = copy.deepcopy(preset_datasets)
        self.preset_models = copy.deepcopy(preset_models)
    
    def get_preset_datasets(self):
        return self.preset_datasets 
    
    def get_preset_models(self):
        return self.preset_models
    
    def start_eval_task(self, user_name, model_path, datasets, model):
        if len(model_path)==0 or len(user_name)==0:
            msg = "model_path/url and user_name are necessary."
            gr.Warning(msg)
            yield {"msg":msg}
        else:
            json_post_params = {
                "user_name": user_name,
                "datasets": datasets,
                "model": model,
                "model_path": model_path
            }
            resp = {"msg":"stop..."}
            yield resp
            try:
                response = requests.post(url=self.submit_api, json=json_post_params, timeout=30)
                print(response.json())
                resp = response.json()['data']
            except Exception as e:
                print(f"exception when call {self.submit_api}")
                traceback.print_exc()
                resp = {"msg": str(e)}
            finally:
                yield resp

    def stop_eval_task(self, task_id, user_name, model_path):
        if len(task_id)==0 or len(model_path)==0 or len(user_name)==0:
            msg = "task_id, model_path/url and user_name are necessary for stop."
            gr.Warning(msg)
            yield {"msg": msg}
        else:
            json_post_params = {
                "id": task_id,
                "user_name": user_name,
                "model_path": model_path
            }
            resp = {"msg": "stop..."}
            yield resp
            try:
                response = requests.post(url=self.stop_api, json=json_post_params, timeout=30)
                resp = response.json()['data']
            except Exception as e:
                print(f"exception when call {self.stop_api}")
                traceback.print_exc()
                resp = {"msg": str(e)}
            finally:
                yield resp
        
    def update_eval_status(self):
        resp = {"msg": ""}
        try:
            response = requests.get(url=self.update_api, timeout=30)
            resp = response.json()['data']
        except Exception as e:
            traceback.print_exc()
            resp = {"msg": str(e)}
        finally:
            return resp

class LLMData(object):
    def __init__(self, preset_models, instruction_folder):
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.preset_models = {m["name"]:m["url"] for m in preset_models}
        self.instruction_folder = os.path.abspath(os.path.join(root_folder, instruction_folder))
        self.system_filename = "system.md"
        self.fewshot_filename = "few_shot.json"

    def get_preset_prompts(self):
        return list(os.listdir(self.instruction_folder))
    
    def get_preset_models(self):
        return self.preset_models
    
    def select_preset_model(self, model, output):
        if model:
            if model in self.preset_models.keys():
                msg = f"url for model {model} found."
                output = append_output_msg(output_msg=output, append_msg=msg)
                url = self.preset_models[model]
            else:
                url = ""
                msg = f"url for model {model} not found."
                gr.Warning(msg)
            output = append_output_msg(output_msg=output, append_msg=msg)
            return url, output
        else:
            return "", output

    def load_instruction_content(self, path:str, output:str):
        if not isinstance(path, str) or len(path)==0:
            __msg = "no prompt is selected."
            gr.Warning(__msg)
            output = append_output_msg(output, __msg)
            return "", "", output
        
        path = os.path.join(self.instruction_folder, path)
        try:
            with open(os.path.join(path, self.system_filename), mode="r") as fp:
                system = "".join(fp.readlines())
            fewshot_path = os.path.join(path, self.fewshot_filename)
            if os.path.exists(fewshot_path):
                with open(fewshot_path, mode="r") as fp:
                    data = json.load(fp)
            else:
                msg = f"fewshot file not found."
                gr.Warning(msg)
                data = []            
            few_shot = json.dumps(data, indent=4)
            
            msg = "load successfully."
            gr.Info(msg)
        except Exception as e:
            system = few_shot = ""
            msg = str(e)
            gr.Warning(msg)
        finally:
            output = append_output_msg(output, msg)
        return system, few_shot, output
    
    def save_instruction_content(self, path:str, system:str, few_shot:str, output:str):
        if not isinstance(path, str) or len(path)==0:
            __msg = "save path not avilable."
            gr.Warning(__msg)
            output = append_output_msg(output, __msg)
            return output
        path = os.path.join(self.instruction_folder, path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        try:
            with open(os.path.join(path, self.system_filename), mode="w") as fp:
                fp.write(system)
            with open(os.path.join(path, self.fewshot_filename), mode="w") as fp:
                content = json.loads(few_shot)
                json.dump(content, fp=fp, indent=4) 
            msg = "save successfully."
            gr.Info(msg)
        except Exception as e:
            msg = str(e)
            gr.Warning(msg)
        finally:
            output = append_output_msg(output, msg)
        return output

    def update_instruction(self):
        return gr.Dropdown(choices=self.get_preset_prompts())

    @staticmethod
    def chat_stream(url, system:str, few_shot:str, msg, params, output):
        chatbot = []
        if not isinstance(url, str) or len(url)==0:
            __msg = "no model url."
            gr.Warning(__msg)
            output = append_output_msg(output, __msg)
            yield chatbot, output
        else:
            try:
                messages = json.loads(few_shot)
                messages = [{"role":"system", "content":system}]+messages
                _, chatbot, system = LLMServer.parse_messages(messages=messages)
                yield chatbot, output
                for resp in LLMDebug.response_wrapper_llm(url, system, msg, chatbot, params, output):
                    chatbot, output = resp
                    yield chatbot, output
            except Exception as e:
                output += str(e)
            finally:
                yield chatbot, output

    @staticmethod
    def get_default_parameters(url, output_msg):
        return LLMDebug.get_default_parameters(url, output_msg)

#### 分析IM数据
class IMDataAnalyse(object):
    def __init__(self, data_path:str):
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dict = self.load_data(os.path.join(root_folder, data_path))

    def load_data(self, path:str)->dict:
        data = pd.read_csv(path, quoting=csv.QUOTE_MINIMAL)
        data_dict = {row["iid"]:json.loads(row["messages"]) for _, row in data.iterrows()}
        self.data_dict = data_dict
        return data_dict
    
    def get_iids(self):
        return list(self.data_dict.keys())
    
    def index(self, iid):
        print(iid)
        return self.data_dict.get(iid, [])
    
    @classmethod
    def messages2history(cls, messages:list):
        messages = copy.deepcopy(messages)
        history = []
        while len(messages)>0:
            m = messages.pop(0)
            if len(messages)==0 or messages[0]["role"]==m["role"]:
                if m["role"]=="骑手":
                    dialog = [None, m["timestamp"]+ "\t" + m["content"]]
                else:
                    dialog = [m["timestamp"]+ "\t" + m["content"], None]
            else:
                
                if m["role"]=="骑手":
                    dialog = [None, m["timestamp"]+ "\t" + m["content"]]
                else: # 站长-骑手
                    m1 = messages.pop(0)
                    dialog = [m["timestamp"]+ "\t" + m["content"], m1["timestamp"]+ "\t" + m1["content"]]
            history.append(dialog)
            yield history


#### IMCopilot
class IMCopilot(object):
    def __init__(self, llm_models:list, embedding_models:list, preset_data:str):
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.llm_models = {l["name"]:l["url"] for l in llm_models}
        self.embedding_models = {l["name"]:l["url"] for l in embedding_models}
        self.preset_data = read_yaml(os.path.join(root_folder, preset_data))

    def get_llm_models(self):
        return list(self.llm_models.keys())
    def get_embedding_models(self):
        return list(self.embedding_models.keys())
    def get_llm_model_url(self, name):
        return self.llm_models.get(name, "")
    def get_embedding_model_url(self, name):
        return self.embedding_models.get(name, "")

    def __semantic_recall(self, query, content:dict, embedding_url, threshold)->dict:
        similarity_api = f"{embedding_url}/api/similarity"
        content_set = list(content.keys())
        params = {
            "query_list1": content_set,
            "query_list2": [query]
        }
        similarity_matrix = requests.post(similarity_api, json=params, timeout=10).json()["data"]["score"]
        similarity_matrix_selected = (np.array(similarity_matrix)>threshold).squeeze()
        content_idx = np.array(list(range(len(content_set))))[similarity_matrix_selected]
        content_result = {content[content_set[idx]]:similarity_matrix[idx][0] for idx in content_idx}
        print(content_result)
        return content_result
    
    def __get_intent(self, query:str, intent:str, embedding_url:str, threshold:float)->dict:
        intent_list = [l.split() for l in intent.split("\n") if len(l.strip())>0]
        intent_dict = {l[0].strip():l[1] for l in intent_list if len(l)>=2}
        return self.__semantic_recall(query, intent_dict, embedding_url, threshold=threshold)

    def __get_knowledge(self, query:str, knowledge:str, embedding_url:str, threshold:float)->dict:
        knowledge_list = [l.split() for l in knowledge.split("\n") if len(l.strip())>0]
        knowledge_dict = {l[0].strip():l[1].strip() for l in knowledge_list if len(l)>=2}
        return self.__semantic_recall(query, knowledge_dict, embedding_url, threshold=threshold)
    
    def __get_tools(self, response, tools:str, embedding_url:str, threshold:float)->dict:
        tool_list = [l.split() for l in tools.split("\n") if len(l.strip())>0]
        tool_dict = {l[1].strip():l[0].strip() for l in tool_list if len(l)>=2}
        print(f"tool query: {response}")
        print(f"tool dict: {tool_dict}")
        return self.__semantic_recall(response, tool_dict, embedding_url, threshold=threshold)

    def __update_context(self, context_manager:dict, intents:dict, threshold:float=0.3):
        cur_weight = 0.6
        if not isinstance(context_manager, dict):
            try:
                context_manager = json.loads(context_manager)
            except Exception as e:
                context_manager = {}
        keys = set(intents.keys()).union(set(context_manager.keys()))
        for k in keys:
            context_manager[k] = context_manager.get(k, 0)*(1-cur_weight)+ intents.get(k, 0) * cur_weight
        context_manager = {k:v for k,v in context_manager.items() if v>threshold}
        return json.dumps(context_manager, indent=4, ensure_ascii=False)
    
    def __construct_system(self, instruct:str, intents:list, knowlesge_this:list):
        intents =["### 意图"] + intents + [""]
        knowlesge_this = ["### 知识库"] + knowlesge_this + [""]
        return "\n".join([instruct]+intents+knowlesge_this)
        
    def generate_stream(self, instruct:str, knowledge:str, intent:str, tools:str, llm_url:str, embedding_url:str, intent_threshold:float, knowledge_threshold:float, tool_threshold:float, history:list, query:str, context_manager:dict, output_msg:str, regenerate=False):
        # 意图识别
        intents = self.__get_intent(query, intent, embedding_url, intent_threshold)
        output_msg += f"intents: {intents}\n"
        # 更新上下文
        context_manager = self.__update_context(context_manager=context_manager, intents=intents)
        yield history, context_manager, output_msg
        # 知识库检索
        knowledge_this = self.__get_knowledge(query, knowledge, embedding_url, knowledge_threshold)
        output_msg += f"knowledge: {knowledge_this}\n"
        yield history, context_manager, output_msg
        # 回复
        if regenerate:
            generate_func = LLMDebug.regenerate_llm
        else:
            generate_func = LLMDebug.response_wrapper_llm
        system=self.__construct_system(instruct, list(intents.keys()), list(knowledge_this.keys()))
        for chatbot, _out_msg in generate_func(llm_url, system, query, history, {}, output_msg):
            yield chatbot, context_manager, _out_msg

        history = chatbot
        output_msg = _out_msg
        # 工具推荐
        tools_this = self.__get_tools(history[-1][-1], tools, embedding_url, tool_threshold)
        if len(tools_this)==0:
            output_msg += f"\nno relative tool."
        else:
            output_msg += str(tools_this)
            sorted_keys = sorted(tools_this.keys(), key=tools_this.get, reverse=True)
            response = "\n".join([f"{k}:{tools_this[k]}" for k in sorted_keys])
            history += [(None, response)]
        yield history, context_manager, output_msg 

class About(object):
    def __init__(self, intro_path:str):
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.intro_path = os.path.join(root_path, intro_path)

    def read_intro(self):
        with open(self.intro_path, mode="r") as fp:
            data = fp.readlines()
        return "".join(data)

#### Functional Call
class LLMFunctionallCall(object):
    def __init__(self, supported_models:list, tools:list):
        self.supported_models = {m["name"]:m["url"] for m in supported_models}
        self.tools = {t["name"]:t["cfg"] for t in tools}
        self.function_map = {}
        self._init_tools(self.tools)

    def _init_tools(self, tools:dict):
        for tool_name, tool_cfg in tools.items():
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            if tool_name in self.function_map:
                print(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)


    def get_model_list(self):
        return list(self.supported_models.keys())
    
    def get_tools(self):
        return list(self.tools.keys())
    
    def call_llm_with_tools(self, url:str, tools:list, system:str, query:str, history:list, output:str, params:dict={}):
        llm_server = LLMServer(url=url)
        messages = [Message(role="system", content=system)]
        for q, a in history:
            if q is not None and len(q)>0:
                messages.append(Message(role="user", content=q))
            if a is not None and len(a)>0:
                messages.append(Message(role="assistant", content=a))
        messages.append(Message(role="user", content=query))
        messages = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=messages,
            functions=[self.function_map[t] for t in tools],
            lang=None,
        )
        messages = [m.model_dump() for m in messages]
        chat_qa = [query, ""]
        response = llm_server.chat_stream(messages, params)
        for chunk in response.iter_lines():
            chunk = json.decode()
            text = json.loads(chunk)['data']['text']
            print(text)
            if chunk.status_code == HTTPStatus.OK:
                chat_qa[-1] = text
                yield history+[chat_qa], output
            else:
                yield history+[chat_qa], append_output_msg(output, f"error with code: {chunk['data']['code']}")
        yield history+[chat_qa], append_output_msg(output, f"call_llm_with_tools success.")


# 将VoiceAnnotation类定义移到main函数之前
# 应该放在其他类定义之后，比如About类之后，main函数之前

class VoiceAnnotation(object):
    def __init__(self, qa_data_path=None, output_folder=None, s3plus_config=None):
        """
        初始化语音标注类
        
        Args:
            qa_data_path: 问题-答案对数据的路径
            output_folder: 音频输出文件夹
            s3plus_config: S3Plus配置，用于上传音频
        """
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 直接指定固定路径
        self.qa_data_path = os.path.join(root_folder, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-peisongpa-llm-dolphin/hadoop-hmart-peisongpa/lijiguo/workspace/banma_llm_base_model/scripts/evaluation/data/voice_annotation/qa_pairs.json")
        self.output_folder = os.path.join(root_folder, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-peisongpa-llm-dolphin/hadoop-hmart-peisongpa/lijiguo/workspace/banma_llm_base_model/scripts/evaluation/data/voice_annotation")
        
        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.qa_data_path), exist_ok=True)
        
        # 如果JSON文件不存在，创建一个示例文件
        if not os.path.exists(self.qa_data_path):
            example_data = [
                {
                    "question": "什么是人工智能？",
                    "answer": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于开发能够模拟人类智能行为的系统。它包括机器学习、自然语言处理、计算机视觉等多个领域，使计算机能够执行通常需要人类智能的任务，如识别语音、图像、做出决策等。"
                },
                {
                    "question": "大语言模型是如何工作的？",
                    "answer": "大语言模型（如GPT、BERT等）基于深度学习中的Transformer架构，通过在海量文本数据上进行预训练来学习语言的统计规律。它们能够理解上下文，生成连贯的文本，回答问题，甚至执行简单的推理。这些模型通过自注意力机制处理输入文本的每个部分，并预测下一个可能的词或句子。"
                },
                {
                    "question": "什么是机器学习？",
                    "answer": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测或决策的算法，而无需明确编程。它包括监督学习、无监督学习和强化学习等方法。在监督学习中，算法从带标签的训练数据中学习；在无监督学习中，算法从无标签数据中发现模式；在强化学习中，算法通过与环境交互并获得反馈来学习。"
                }
            ]
            with open(self.qa_data_path, 'w', encoding='utf-8') as f:
                json.dump(example_data, f, ensure_ascii=False, indent=2)
        
        # 加载问题-答案对数据
        self.qa_pairs = self.load_qa_pairs()
        
        # 生成唯一的用户标识符，用于区分不同用户的录音
        self.user_id = secrets.token_hex(8)
        
        # 创建用户专属的临时目录
        self.user_temp_dir = os.path.join(tempfile.gettempdir(), f"voice_annotation_{self.user_id}")
        os.makedirs(self.user_temp_dir, exist_ok=True)
        
        # 创建用户状态文件，用于存储当前索引
        self.user_state_file = os.path.join(self.user_temp_dir, "state.json")
        if os.path.exists(self.user_state_file):
            with open(self.user_state_file, 'r') as f:
                state = json.load(f)
                self.current_index = state.get("current_index", 0)
        else:
            self.current_index = 0
            self._save_user_state()
        
        # 初始化S3Plus客户端（如果提供了配置）
        self.s3plus = None
        if s3plus_config:
            self.s3plus = S3Plus(**s3plus_config)
    
    def _save_user_state(self):
        """保存用户状态"""
        with open(self.user_state_file, 'w') as f:
            json.dump({"current_index": self.current_index}, f)
    
    def load_qa_pairs(self):
        """加载问题-答案对数据"""
        try:
            if self.qa_data_path.endswith('.csv'):
                df = pd.read_csv(self.qa_data_path)
                return [(row['接听人'], row['招聘者']) for _, row in df.iterrows()]
            elif self.qa_data_path.endswith('.json'):
                with open(self.qa_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [(item['接听人'], item['招聘者']) for item in data]
            else:
                return [("示例问题1", "示例答案1"), ("示例问题2", "示例答案2")]
        except Exception as e:
            print(f"加载问题-答案对数据失败: {e}")
            return [("示例问题1", "示例答案1"), ("示例问题2", "示例答案2")]
    
    def get_current_qa(self):
        """获取当前问题-答案对"""
        if not self.qa_pairs:
            return "无数据", "无数据", f"0/0", self.current_index  # 返回当前索引
        
        question, answer = self.qa_pairs[self.current_index]
        progress = f"{self.current_index + 1}/{len(self.qa_pairs)}"
        return question, answer, progress, self.current_index  # 返回当前索引
    
    def next_qa(self, current_index=None):
        """切换到下一个问题-答案对"""
        if not self.qa_pairs:
            return "无数据", "无数据", f"0/0", self.current_index
        
        # 如果提供了当前索引，先同步实例的索引
        if current_index is not None and current_index != self.current_index:
            self.current_index = current_index
        
        if self.current_index < len(self.qa_pairs) - 1:
            self.current_index += 1
            self._save_user_state()
        
        return self.get_current_qa()
    
    def prev_qa(self, current_index=None):
        """切换到上一个问题-答案对"""
        if not self.qa_pairs:
            return "无数据", "无数据", f"0/0", self.current_index
        
        # 如果提供了当前索引，先同步实例的索引
        if current_index is not None and current_index != self.current_index:
            self.current_index = current_index
        
        if self.current_index > 0:
            self.current_index -= 1
            self._save_user_state()
        
        return self.get_current_qa()
    
    def save_audio(self, audio, output_msg, index=None):
        """保存录制的音频 - 使用流式处理方法"""
        if audio is None:  # 修改判断条件，只检查是否为None
            output_msg = append_output_msg(output_msg, "没有录音数据")
            return None, output_msg
        
        try:
            # 获取当前问题-答案对
            if not self.qa_pairs:
                output_msg = append_output_msg(output_msg, "无数据可标注")
                return None, output_msg
            
            # 使用传入的索引，如果没有传入则使用实例的current_index
            idx = index if index is not None else self.current_index
            question, answer = self.qa_pairs[idx]
            
            # 生成文件名（使用问题的前20个字符作为文件名的一部分）
            question_part = re.sub(r'[^\w\s]', '', question[:20]).strip().replace(' ', '_')
            
            # 检查是否已存在该问答对的录音，如果存在则删除
            existing_files = []
            if os.path.exists(self.output_folder):
                for filename in os.listdir(self.output_folder):
                    if filename.startswith(f"qa_{idx}_") and filename.endswith(".wav"):
                        existing_files.append(os.path.join(self.output_folder, filename))
            
            # 删除已存在的录音文件
            for file_path in existing_files:
                try:
                    os.remove(file_path)
                    output_msg = append_output_msg(output_msg, f"已删除旧录音: {os.path.basename(file_path)}")
                except Exception as e:
                    output_msg = append_output_msg(output_msg, f"删除旧录音失败: {str(e)}")
            
            # 添加用户ID和时间戳，确保文件名唯一
            filename = f"qa_{idx}_{question_part}_{self.user_id}_{int(time.time())}.wav"
            filepath = os.path.join(self.output_folder, filename)
            
            # 使用文件锁保存音频
            # 创建一个锁文件
            lock_file = os.path.join(self.output_folder, f".lock_{idx}")
            
            with open(lock_file, 'w') as f:
                # 获取独占锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # 解包音频数据
                    sr, data = audio
                    
                    # 检查数据类型并确保它是适合处理的格式
                    if isinstance(data, np.ndarray):
                        # 记录音频信息以便调试
                        output_msg = append_output_msg(output_msg, f"音频信息: 采样率={sr}Hz, 长度={len(data)}样本, 时长={len(data)/sr:.2f}秒")
                        
                        # 确保数据类型正确
                        if data.dtype != np.int16 and data.dtype != np.float32:
                            data = data.astype(np.float32)
                        
                        # 降低采样率到8kHz以减小文件大小
                        target_sr = 44100
                        if sr > target_sr:
                            output_msg = append_output_msg(output_msg, f"降低采样率从 {sr}Hz 到 {target_sr}Hz")
                            
                            # 使用流式处理方法降低采样率
                            # 将数据分成多个块进行处理，避免一次性加载整个数组
                            from scipy import signal
                            
                            # 创建一个临时文件用于流式写入
                            import tempfile
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            temp_file.close()
                            
                            try:
                                import soundfile as sf
                                import math
                                
                                # 计算重采样后的总长度
                                new_length = int(len(data) * target_sr / sr)
                                
                                # 分块大小 - 每次处理约5秒的音频
                                chunk_size = min(sr * 5, len(data))
                                chunks = math.ceil(len(data) / chunk_size)
                                
                                output_msg = append_output_msg(output_msg, f"分块处理音频: {chunks}个块")
                                
                                # 创建输出文件
                                with sf.SoundFile(temp_file.name, 'w', target_sr, 1, 'PCM_16') as outfile:
                                    for i in range(chunks):
                                        start = i * chunk_size
                                        end = min(start + chunk_size, len(data))
                                        
                                        # 获取当前块
                                        chunk = data[start:end]
                                        
                                        # 重采样当前块
                                        chunk_resampled = signal.resample(chunk, int(len(chunk) * target_sr / sr))
                                        
                                        # 写入重采样后的块
                                        outfile.write(chunk_resampled)
                                
                                # 复制临时文件到目标位置
                                import shutil
                                shutil.copy2(temp_file.name, filepath)
                                
                                output_msg = append_output_msg(output_msg, f"音频已保存为WAV: {filepath}")
                                
                                # 读取处理后的文件，以便返回正确的格式给Gradio
                                processed_data, processed_sr = sf.read(filepath)
                                
                            finally:
                                # 删除临时文件
                                try:
                                    os.unlink(temp_file.name)
                                except:
                                    pass
                        else:
                            # 如果采样率已经很低，直接保存
                            import soundfile as sf
                            sf.write(filepath, data, sr, format='WAV')
                            output_msg = append_output_msg(output_msg, f"音频已保存为WAV: {filepath}")
                            
                            # 使用原始数据和采样率
                            processed_data, processed_sr = data, sr
                        
                        # 如果配置了S3Plus，则上传到S3
                        if self.s3plus:
                            s3_key = f"voice_annotation/{os.path.basename(filepath)}"
                            self.s3plus.upload_file(filepath, s3_key)
                            output_msg = append_output_msg(output_msg, f"音频已保存并上传至S3: {s3_key}")
                        else:
                            output_msg = append_output_msg(output_msg, f"音频已保存至本地: {filepath}")
                        
                        # 返回处理后的音频数据和采样率，而不是文件路径
                        # 这样Gradio可以正确显示音频时长
                        return (processed_sr, processed_data), output_msg
                    else:
                        output_msg = append_output_msg(output_msg, f"音频数据格式不正确: {type(data)}")
                        return None, output_msg
                finally:
                    # 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            # 尝试删除锁文件
            try:
                os.remove(lock_file)
            except:
                pass
            
            return (processed_sr, processed_data), output_msg
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            output_msg = append_output_msg(output_msg, f"保存音频失败: {str(e)}\n{error_details}")
            return None, output_msg
    
    def get_recorded_indices(self):
        """获取已经录制的问答对索引"""
        try:
            recorded_indices = set()
            # 检查输出文件夹中的文件
            if os.path.exists(self.output_folder):
                for filename in os.listdir(self.output_folder):
                    if filename.startswith("qa_") and filename.endswith(".wav"):
                        # 从文件名中提取索引
                        parts = filename.split("_")
                        if len(parts) > 1:
                            try:
                                idx = int(parts[1])
                                recorded_indices.add(idx)
                            except ValueError:
                                continue
            return recorded_indices
        except Exception as e:
            print(f"获取已录制索引失败: {e}")
            return set()
    
    def get_missing_recordings(self):
        """获取缺失的录音"""
        if not self.qa_pairs:
            return []
        
        recorded_indices = self.get_recorded_indices()
        missing_indices = []
        
        # 检查当前索引之前的所有问答对
        for i in range(self.current_index):
            if i not in recorded_indices:
                missing_indices.append(i)
        
        return missing_indices
    
    def get_missing_recordings_info(self):
        """获取缺失录音的信息"""
        missing_indices = self.get_missing_recordings()
        if not missing_indices:
            return "当前没有缺失的录音"
        
        missing_info = f"发现 {len(missing_indices)} 个缺失的录音: " + ", ".join([f"#{i+1}" for i in missing_indices])
        return missing_info
    
    def jump_to_index(self, index):
        """跳转到指定索引的问答对"""
        if not self.qa_pairs:
            return "无数据", "无数据", f"0/0", self.current_index  # 返回当前索引
        
        try:
            index = int(index)
            if 0 <= index < len(self.qa_pairs):
                self.current_index = index
                self._save_user_state()
                return self.get_current_qa()
            else:
                return self.get_current_qa()
        except:
            return self.get_current_qa()
    
    def has_recording(self, index=None):
        """检查指定索引的问答对是否有录音"""
        if index is None:
            index = self.current_index
            
        # 检查输出文件夹中是否有对应索引的录音文件
        if os.path.exists(self.output_folder):
            for filename in os.listdir(self.output_folder):
                if filename.startswith(f"qa_{index}_") and filename.endswith(".wav"):
                    return True
        return False
    
    def get_recording_path(self, index=None):
        """获取指定索引的问答对的录音文件路径"""
        if index is None:
            index = self.current_index
            
        # 查找对应索引的录音文件
        if os.path.exists(self.output_folder):
            matching_files = []
            for filename in os.listdir(self.output_folder):
                if filename.startswith(f"qa_{index}_") and filename.endswith(".wav"):
                    matching_files.append(os.path.join(self.output_folder, filename))
            
            # 如果有多个匹配的文件，返回最新的一个
            if matching_files:
                return sorted(matching_files, key=os.path.getmtime, reverse=True)[0]
        
        return None
    
    def get_current_recording_status(self):
        """获取当前问答对的录音状态和路径"""
        has_recording = self.has_recording()
        recording_path = self.get_recording_path() if has_recording else None
        status = "已录制" if has_recording else "未录制"
        return has_recording, recording_path, status

class VoiceRank(object):
    def __init__(self, rank_data_path=None, output_folder=None):
        """
        初始化语音排名类
        """
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 直接指定固定路径
        self.rank_data_path = os.path.join(root_folder, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-peisongpa-llm-dolphin/hadoop-hmart-peisongpa/lijiguo/workspace/banma_llm_base_model/scripts/evaluation/data/voice_rank/ttzq_ffzlmerged_output_with_new_paths_cleaned_shuffled.json")
        self.output_folder = os.path.join(root_folder, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-peisongpa-llm-dolphin/hadoop-hmart-peisongpa/lijiguo/workspace/banma_llm_base_model/scripts/evaluation/data/voice_rank")
        self.result_path = os.path.join(self.output_folder, "rank_result.json")
        
        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 加载排名数据
        self.rank_data = self.load_rank_data()
        # 移除实例变量 current_group，改为在会话中存储
        self.total_groups = len(self.rank_data)
        
        # 初始化结果数据
        self.load_or_init_results()
    
    def load_rank_data(self):
        """加载排名数据"""
        try:
            if os.path.exists(self.rank_data_path):
                with open(self.rank_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            else:
                print(f"排名数据文件不存在: {self.rank_data_path}")
                return {}
        except Exception as e:
            print(f"加载排名数据失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def load_or_init_results(self):
        """加载或初始化结果数据"""
        try:
            if os.path.exists(self.result_path):
                with open(self.result_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                print(f"已加载排名结果: {len(self.results)} 条记录")
            else:
                self.results = {}
                print("排名结果文件不存在，初始化为空")
        except Exception as e:
            print(f"加载结果数据失败: {e}")
            import traceback
            traceback.print_exc()
            self.results = {}
    
    def get_group(self, group_index):
        """获取指定组的问题和回答"""
        if not self.rank_data or len(self.rank_data) == 0:
            return "无数据", None, "无数据", None, "无数据", None, f"0/0", None, None, None
        
        # 确保索引在有效范围内
        group_index = max(0, min(group_index, self.total_groups - 1))
        
        # 获取指定组的数据
        group_id = str(group_index)
        if group_id not in self.rank_data:
            return "无数据", None, "无数据", None, "无数据", None, f"0/0", None, None, None
        
        group_data = self.rank_data[group_id]
        user_question = group_data.get("user", "无问题")
        
        # 获取模型回答
        models_data = group_data.get("models", {})
        model_names = list(models_data.keys())
        
        # 确保至少有3个模型回答，不足的用空字符串和None补齐
        model_contents = []
        audio_paths = []
        
        for i in range(min(3, len(model_names))):
            model_name = model_names[i]
            model_data = models_data[model_name]
            model_contents.append(f"{model_name}: {model_data.get('content', '')}")
            audio_paths.append(model_data.get('audio_path'))
        
        # 补齐到3个
        while len(model_contents) < 3:
            model_contents.append("")
            audio_paths.append(None)
        
        progress = f"{group_index + 1}/{self.total_groups}"
        
        # 检查是否已有标注结果
        ranks = [None, None, None]
        notes = ["", "", ""]
        
        if group_id in self.results:
            result_data = self.results[group_id]
            models_result = result_data.get("models", {})
            
            for i, model_name in enumerate(model_names[:3]):
                if model_name in models_result:
                    model_result = models_result[model_name]
                    rank_value = model_result.get("rank")
                    
                    # 将数字转换为前端选项格式
                    if rank_value == 1:
                        ranks[i] = "1 (最口语化)"
                    elif rank_value == 2:
                        ranks[i] = "2"
                    elif rank_value == 3:
                        ranks[i] = "3（最不口语化）"
                    
                    notes[i] = model_result.get("note", "")
        
        # 返回问题、3个模型回答和对应的音频路径，以及进度和已有的排名和备注
        return (
            user_question,
            model_contents[0], audio_paths[0], 
            model_contents[1], audio_paths[1], 
            model_contents[2], audio_paths[2],
            progress,
            ranks[0], ranks[1], ranks[2],
            notes[0], notes[1], notes[2]
        )
    
    def next_group(self, current_index):
        """切换到下一组"""
        if not self.rank_data or len(self.rank_data) == 0:
            return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        next_index = current_index + 1 if current_index < self.total_groups - 1 else current_index
        results = self.get_group(next_index)
        return (next_index,) + results
    
    def prev_group(self, current_index):
        """切换到上一组"""
        if not self.rank_data or len(self.rank_data) == 0:
            return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        prev_index = current_index - 1 if current_index > 0 else 0
        results = self.get_group(prev_index)
        return (prev_index,) + results
    
    def save_rank(self, rank1, rank2, rank3, rank4, rank5, note1, note2, note3, output_msg, group_index=None):
        """保存排名结果和备注"""
        try:
            # 获取当前组的ID
            group_id = str(group_index)
            
            # 创建排名数据
            ranks = [rank1, rank2, rank3]
            notes = [note1, note2, note3]
            
            # 检查排名是否有效（1-3，不重复）
            valid_ranks = [r for r in ranks if r in [1, 2, 3]]
            if len(valid_ranks) != len(set(valid_ranks)) or len(valid_ranks) != 3:
                output_msg = append_output_msg(output_msg, "排名无效，请确保每个回答的排名是1-3之间的不重复数字")
                return output_msg
            
            # 获取当前组的数据
            group_data = self.rank_data[group_id]
            
            # 获取模型名称
            models_data = group_data.get("models", {})
            model_names = list(models_data.keys())[:3]
            
            # 提取用户问题
            user_question = group_data.get("user", "多轮对话")
            
            # 保存排名结果和备注
            self.results[group_id] = {
                "user_question": user_question,
                "models": {
                    model_names[i]: {
                        "rank": ranks[i],
                        "note": notes[i] if i < len(notes) else ""
                    } for i in range(min(len(model_names), 3))
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 使用文件锁写入文件
            with open(self.result_path, 'w+', encoding='utf-8') as f:
                # 获取独占写锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # 如果文件已存在且有内容，先读取现有内容
                    f.seek(0)
                    content = f.read().strip()
                    if content:
                        try:
                            existing_results = json.loads(content)
                            # 合并现有结果和新结果
                            existing_results.update(self.results)
                            self.results = existing_results
                        except json.JSONDecodeError:
                            # 如果文件内容不是有效的JSON，则使用当前结果
                            pass
                    
                    # 写入更新后的结果
                    f.seek(0)
                    f.truncate()
                    json.dump(self.results, f, ensure_ascii=False, indent=2)
                finally:
                    # 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            output_msg = append_output_msg(output_msg, f"成功保存第 {group_index + 1} 组的排名结果和备注")
            return output_msg
        except Exception as e:
            output_msg = append_output_msg(output_msg, f"保存排名失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息到控制台
            return output_msg

    def get_missing_rankings(self, current_index):
        """获取缺失的排名组"""
        if not self.rank_data:
            return "无数据可排名", []
        
        # 获取已经排名的组
        ranked_groups = set(self.results.keys())
        
        # 获取所有组
        all_groups = set(self.rank_data.keys())
        
        # 获取当前组之前的所有组
        missing_groups = []
        for i in range(current_index):
            group_id = str(i)
            if group_id not in ranked_groups and group_id in self.rank_data:
                missing_groups.append(i)
        
        if not missing_groups:
            return "当前没有缺失的排名", []
        
        missing_info = f"发现 {len(missing_groups)} 个缺失的排名: " + ", ".join([f"#{i+1}" for i in missing_groups])
        return missing_info, missing_groups
    
    def jump_to_group(self, group_index):
        """跳转到指定组"""
        if not self.rank_data:
            return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        try:
            group_index = int(group_index)
            if 0 <= group_index < self.total_groups:
                results = self.get_group(group_index)
                return (group_index,) + results
            else:
                return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        except Exception as e:
            print(f"跳转失败: {e}")
            return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"

# 添加 VoiceRankMultiple 类，与 VoiceRank 类似但使用不同的数据路径
class VoiceRankMultiple(object):
    def __init__(self, rank_data_path=None, output_folder=None):
        """
        初始化语音排名类 - 多轮对话版本
        """
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 直接指定固定路径 - 使用多轮对话数据
        self.rank_data_path = os.path.join(root_folder, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-peisongpa-llm-dolphin/hadoop-hmart-peisongpa/lijiguo/workspace/banma_llm_base_model/scripts/evaluation/data/voice_rank/dldh_merged_output_with_new_paths_reformatted_output_cleaned.json")
        self.output_folder = os.path.join(root_folder, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-peisongpa-llm-dolphin/hadoop-hmart-peisongpa/lijiguo/workspace/banma_llm_base_model/scripts/evaluation/data/voice_rank")
        self.result_path = os.path.join(self.output_folder, "rank_result_multiple.json")
        
        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 加载排名数据
        self.rank_data = self.load_rank_data()
        self.current_group = 0
        self.total_groups = len(self.rank_data)
        
        # 初始化结果数据
        self.load_or_init_results()
    
    def load_rank_data(self):
        """加载排名数据"""
        try:
            if os.path.exists(self.rank_data_path):
                with open(self.rank_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            else:
                print(f"排名数据文件不存在: {self.rank_data_path}")
                return {}
        except Exception as e:
            print(f"加载排名数据失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def load_or_init_results(self):
        """加载或初始化结果数据"""
        try:
            if os.path.exists(self.result_path):
                with open(self.result_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                print(f"已加载排名结果: {len(self.results)} 条记录")
            else:
                self.results = {}
                print("排名结果文件不存在，初始化为空")
        except Exception as e:
            print(f"加载结果数据失败: {e}")
            import traceback
            traceback.print_exc()
            self.results = {}
    
    def get_current_group(self):
        """获取当前组的问题和回答"""
        if not self.rank_data or len(self.rank_data) == 0:
            return "无数据", None, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        # 获取当前组的数据
        group_id = str(self.current_group)
        if group_id not in self.rank_data:
            return "无数据", None, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        group_data = self.rank_data[group_id]
        
        # 从模型回答中提取对话内容
        models_data = group_data  # 在dldh.json中，组数据直接就是模型数据
        model_names = list(models_data.keys())
        
        # 提取第一个模型的对话内容中的用户问题
        # 假设所有模型的对话都包含相同的用户问题
        if len(model_names) > 0:
            first_model_content = models_data[model_names[0]].get('content', '')
            # 尝试从内容中提取用户问题
            user_question = "多轮对话"
            if 'user:' in first_model_content:
                # 提取第一个user:后面的内容作为问题
                try:
                    user_question = first_model_content.split('user:')[1].split('assistant:')[0].strip()
                except:
                    pass
        else:
            user_question = "无问题"
        
        # 确保至少有3个模型回答，不足的用空字符串和None补齐
        model_contents = []
        audio_paths = []
        
        for i in range(min(3, len(model_names))):
            model_name = model_names[i]
            model_data = models_data[model_name]
            model_contents.append(f"{model_name}: {model_data.get('content', '')}")
            audio_paths.append(model_data.get('audio_path'))
        
        # 补齐到3个
        while len(model_contents) < 3:
            model_contents.append("")
            audio_paths.append(None)
        
        progress = f"{self.current_group + 1}/{self.total_groups}"
        
        # 检查是否已有标注结果
        ranks = [None, None, None]
        notes = ["", "", ""]
        
        if group_id in self.results:
            result_data = self.results[group_id]
            models_result = result_data.get("models", {})
            
            for i, model_name in enumerate(model_names[:3]):
                if model_name in models_result:
                    model_result = models_result[model_name]
                    rank_value = model_result.get("rank")
                    
                    # 将数字转换为前端选项格式
                    if rank_value == 1:
                        ranks[i] = "1 (最口语化)"
                    elif rank_value == 2:
                        ranks[i] = "2"
                    elif rank_value == 3:
                        ranks[i] = "3（最不口语化）"
                    
                    notes[i] = model_result.get("note", "")
        
        # 返回问题、3个模型回答和对应的音频路径，以及进度和已有的排名和备注
        return (
            user_question,
            model_contents[0], audio_paths[0], 
            model_contents[1], audio_paths[1], 
            model_contents[2], audio_paths[2],
            progress,
            ranks[0], ranks[1], ranks[2],
            notes[0], notes[1], notes[2]
        )
    
    def next_group(self, current_index):
        """切换到下一组"""
        if not self.rank_data or len(self.rank_data) == 0:
            return current_index, "无数据", None, "无数据", None, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        next_index = current_index + 1 if current_index < self.total_groups - 1 else current_index
        results = self.get_group(next_index)
        return (next_index,) + results
    
    def prev_group(self, current_index):
        """切换到上一组"""
        if not self.rank_data or len(self.rank_data) == 0:
            return current_index, "无数据", None, "无数据", None, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        prev_index = current_index - 1 if current_index > 0 else 0
        results = self.get_group(prev_index)
        return (prev_index,) + results
    
    def save_rank(self, rank1, rank2, rank3, rank4, rank5, note1, note2, note3, output_msg, group_index=None):
        """保存排名结果和备注"""
        try:
            # 获取当前组的ID - 这里需要从参数中获取
            group_id = str(group_index)  # 这行有问题
            
            # 创建排名数据
            ranks = [rank1, rank2, rank3]
            notes = [note1, note2, note3]
            
            # 检查排名是否有效（1-3，不重复）
            valid_ranks = [r for r in ranks if r in [1, 2, 3]]
            if len(valid_ranks) != len(set(valid_ranks)) or len(valid_ranks) != 3:
                output_msg = append_output_msg(output_msg, "排名无效，请确保每个回答的排名是1-3之间的不重复数字")
                return output_msg
            
            # 获取当前组的模型名称
            group_data = self.rank_data[group_id]
            model_names = list(group_data.keys())[:3]
            
            # 提取第一个模型的对话内容中的用户问题
            user_question = "多轮对话"
            if len(model_names) > 0:
                first_model_content = group_data[model_names[0]].get('content', '')
                if 'user:' in first_model_content:
                    try:
                        user_question = first_model_content.split('user:')[1].split('assistant:')[0].strip()
                    except:
                        pass
            
            # 保存排名结果和备注
            self.results[group_id] = {
                "user_question": user_question,
                "models": {
                    model_names[i]: {
                        "rank": ranks[i],
                        "note": notes[i] if i < len(notes) else ""
                    } for i in range(min(len(model_names), 3))
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 使用文件锁写入文件
            with open(self.result_path, 'w+', encoding='utf-8') as f:
                # 获取独占写锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # 如果文件已存在且有内容，先读取现有内容
                    f.seek(0)
                    content = f.read().strip()
                    if content:
                        try:
                            existing_results = json.loads(content)
                            # 合并现有结果和新结果
                            existing_results.update(self.results)
                            self.results = existing_results
                        except json.JSONDecodeError:
                            # 如果文件内容不是有效的JSON，则使用当前结果
                            pass
                    
                    # 写入更新后的结果
                    f.seek(0)
                    f.truncate()
                    json.dump(self.results, f, ensure_ascii=False, indent=2)
                finally:
                    # 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            output_msg = append_output_msg(output_msg, f"成功保存第 {group_index + 1} 组的排名结果和备注")
            return output_msg
        except Exception as e:
            output_msg = append_output_msg(output_msg, f"保存排名失败: {str(e)}")
            return output_msg

    def get_missing_rankings(self, current_index):
        """获取缺失的排名组"""
        if not self.rank_data:
            return "无数据可排名", []
        
        # 获取已经排名的组
        ranked_groups = set(self.results.keys())
        
        # 获取所有组
        all_groups = set(self.rank_data.keys())
        
        # 获取当前组之前的所有组
        missing_groups = []
        for i in range(current_index):
            group_id = str(i)
            if group_id not in ranked_groups and group_id in self.rank_data:
                missing_groups.append(i)
        
        if not missing_groups:
            return "当前没有缺失的排名", []
        
        missing_info = f"发现 {len(missing_groups)} 个缺失的排名: " + ", ".join([f"#{i+1}" for i in missing_groups])
        return missing_info, missing_groups
    
    def jump_to_group(self, group_index):
        """跳转到指定组"""
        if not self.rank_data:
            return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        
        try:
            group_index = int(group_index)
            if 0 <= group_index < self.total_groups:
                results = self.get_group(group_index)
                return (group_index,) + results
            else:
                return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"
        except Exception as e:
            print(f"跳转失败: {e}")
            return 0, "无数据", None, "无数据", None, "无数据", None, f"0/0"

    def get_group(self, group_index):
        """获取指定组的问题和回答"""
        if not self.rank_data or len(self.rank_data) == 0:
            return "无数据", None, "无数据", None, "无数据", None, f"0/0", None, None, None, None, None, None
        
        # 确保索引在有效范围内
        group_index = max(0, min(group_index, self.total_groups - 1))
        
        # 获取指定组的数据
        group_id = str(group_index)
        if group_id not in self.rank_data:
            return "无数据", None, "无数据", None, "无数据", None, f"0/0", None, None, None, None, None, None
        
        # 从模型回答中提取对话内容
        group_data = self.rank_data[group_id]
        model_names = list(group_data.keys())
        
        # 提取第一个模型的对话内容中的用户问题
        # 假设所有模型的对话都包含相同的用户问题
        if len(model_names) > 0:
            first_model_content = group_data[model_names[0]].get('content', '')
            # 尝试从内容中提取用户问题
            user_question = "多轮对话"
            if 'user:' in first_model_content:
                # 提取第一个user:后面的内容作为问题
                try:
                    user_question = first_model_content.split('user:')[1].split('assistant:')[0].strip()
                except:
                    pass
        else:
            user_question = "无问题"
        
        # 确保至少有3个模型回答，不足的用空字符串和None补齐
        model_contents = []
        audio_paths = []
        
        for i in range(min(3, len(model_names))):
            model_name = model_names[i]
            model_data = group_data[model_name]
            model_contents.append(f"{model_name}: {model_data.get('content', '')}")
            audio_paths.append(model_data.get('audio_path'))
        
        # 补齐到3个
        while len(model_contents) < 3:
            model_contents.append("")
            audio_paths.append(None)
        
        progress = f"{group_index + 1}/{self.total_groups}"
        
        # 检查是否已有标注结果
        ranks = [None, None, None]
        notes = ["", "", ""]
        
        if group_id in self.results:
            result_data = self.results[group_id]
            models_result = result_data.get("models", {})
            
            for i, model_name in enumerate(model_names[:3]):
                if model_name in models_result:
                    model_result = models_result[model_name]
                    rank_value = model_result.get("rank")
                    
                    # 将数字转换为前端选项格式
                    if rank_value == 1:
                        ranks[i] = "1 (最口语化)"
                    elif rank_value == 2:
                        ranks[i] = "2"
                    elif rank_value == 3:
                        ranks[i] = "3（最不口语化）"
                    
                    notes[i] = model_result.get("note", "")
        
        # 返回问题、3个模型回答和对应的音频路径，以及进度和已有的排名和备注
        return (
            user_question,
            model_contents[0], audio_paths[0], 
            model_contents[1], audio_paths[1], 
            model_contents[2], audio_paths[2],
            progress,
            ranks[0], ranks[1], ranks[2],
            notes[0], notes[1], notes[2]
        )

def main(args):
    cfg = read_yaml(args.cfg_path)
    llm_debug_instance = LLMDebug(**cfg["debug"])
    llm_deployment_instance = LLMDeployment(**cfg["deployment"])
    llm_evaluation_instance = LLMEvalution(**cfg["evaluation"])
    bm_llm_evaluation_instance = BMLLMEvaluation(**cfg["bm_evaluation"])
    vlm_server_instance = VLMServer(**cfg["vlm"])
    llm_arena_instance = LLMArena(**cfg["arena"])
    llm_data_instance = LLMData(**cfg["data"])
    im_copilot_instance = IMCopilot(**cfg["im_copilot"])
    about_instance = About(**cfg["about"])
    
    # 初始化语音标注实例
    voice_annotation_instance = VoiceAnnotation()

    # 在main函数中，初始化语音排名实例
    voice_rank_instance = VoiceRank()

    # 在main函数中，初始化语音排名多轮对话实例
    voice_rank_multiple_instance = VoiceRankMultiple()

    def update_instance(cfg_path):
        def __inner__():
            print("update_instance...")
            global llm_debug_instance, llm_evaluation_instance, llm_deployment_instance
            global vlm_server_instance, llm_arena_instance, about_instance
            cfg = {}
            try:
                cfg = read_yaml(cfg_path)
                llm_debug_instance = LLMDebug(**cfg["debug"])
                llm_deployment_instance = LLMDeployment(**cfg["deployment"])
                llm_evaluation_instance = LLMEvalution(**cfg["evaluation"])
                vlm_server_instance = VLMServer(**cfg["vlm"])
                llm_arena_instance = LLMArena(**cfg["arena"])
                about_instance = About(**cfg["about"])
            except Exception as e:
                print(e)
            finally:
                return cfg
        return __inner__

    # 创建 Gradio 界面
    with gr.Blocks(title="LLM Evaluation") as demo:
        with gr.Tab("LLM Arena"):
            with gr.Tab("评测"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            with gr.Column(scale=2):
                                llm_arena_chatbot_left_info = gr.State(value={"name": "", "url":""})
                                llm_arena_chatbot_left = gr.Chatbot(label="LLM Bot1")
                            with gr.Column(scale=2):
                                llm_arena_chatbot_right_info = gr.State(value={"name": "", "url":""})
                                llm_arena_chatbot_right = gr.Chatbot(label="LLM Bot2")
                with gr.Group():
                    with gr.Row():
                        llm_arena_input = gr.Textbox(label="Intput", interactive=False, visible=True, max_lines=2)
                    with gr.Row():
                        llm_arena_show_answer_button = gr.Button(value="显示参考回答", visible=False)
                    with gr.Row():
                        llm_arena_answer = gr.Textbox(label="Answer", interactive=False, visible=False, max_lines=2)
                    with gr.Row():
                        llm_arena_output = gr.Textbox(label="Output", interactive=False, visible=True, max_lines=2)
                    with gr.Row():
                        llm_arena_start_button = gr.Button(value="开始评测(Start)", visible=True)
                        llm_arena_both_button = gr.Button(value="🤝都可以(Tie)", visible=False)
                        llm_arena_left_button = gr.Button(value="👈【左边】更好(Left Better)", visible=False)
                        llm_arena_right_button = gr.Button(value="👉【右边】更好(Right Better)", visible=False)
                        llm_arena_bad_button = gr.Button(value="👎都很差(Both Bad)", visible=False)
                    llm_arena_state = gr.Radio(choices=["tie", "left_better", "right_better", "bad"], interactive=False, visible=False)

            def llm_arena_func_start():
                show_answer_button = gr.Button(visible=True)
                start_button = gr.Button(visible=False)
                both_button = gr.Button(visible=True)
                left_button = gr.Button(visible=True)
                right_button = gr.Button(visible=True)
                bad_button = gr.Button(visible=True)
                return show_answer_button, start_button, both_button, left_button, right_button, bad_button
            
            def llm_arena_func_chat(llm_arena_chatbot_info, msg, output):
                url = llm_arena_chatbot_info["url"]
                for chatbot, output_msg in LLMDebug.response_wrapper_llm(url=url, system="", msg=msg, chatbot=[], params={}, output_msg=output):
                    yield chatbot, output_msg
            
            def llm_arena_func_save_result_and_get_question(result, llm_arena_chatbot1_info, chatbot1, llm_arena_chatbot2_info, chatbot2):
                if len(result)>0:
                    llm_arena_instance.save_result(result=result, bot1_info=llm_arena_chatbot1_info, chatbot1=chatbot1, bot2_info=llm_arena_chatbot2_info, chatbot2=chatbot2)
                chatbot1, chatbot2 = [], []
                msg, ans = llm_arena_instance.get_question()
                bot1_info, bot2_info = llm_arena_instance.get_paired_models()
                return bot1_info, bot2_info, msg, ans, chatbot1, chatbot2
            
            def llm_arena_show_text():
                return gr.Button(visible=False), gr.Textbox(visible=True)
            
            def llm_arena_hide_text():
                return gr.Button(visible=True), gr.Textbox(visible=False)
            
            llm_arena_show_answer_button.click(llm_arena_show_text, [], [llm_arena_show_answer_button, llm_arena_answer])

            llm_arena_start_button.click(llm_arena_func_start, [], [llm_arena_show_answer_button, llm_arena_start_button, llm_arena_both_button, llm_arena_left_button, llm_arena_right_button, llm_arena_bad_button]).then(
                llm_arena_instance.get_question, [], [llm_arena_input, llm_arena_answer], show_progress=True).then(
                llm_arena_instance.get_paired_models, [], [llm_arena_chatbot_left_info, llm_arena_chatbot_right_info]).then(
                llm_arena_func_chat, [llm_arena_chatbot_left_info, llm_arena_input, llm_arena_output], [llm_arena_chatbot_left, llm_arena_output], show_progress=True).then(
                llm_arena_func_chat, [llm_arena_chatbot_right_info, llm_arena_input, llm_arena_output], [llm_arena_chatbot_right, llm_arena_output], show_progress=True)
            
            llm_arena_state.select(llm_arena_hide_text, [], [llm_arena_show_answer_button, llm_arena_answer]).then(common_clear, llm_arena_output, llm_arena_output).then(
                llm_arena_func_save_result_and_get_question, [llm_arena_state, llm_arena_chatbot_left_info, llm_arena_chatbot_left, llm_arena_chatbot_right_info, llm_arena_chatbot_right],
                                        [llm_arena_chatbot_left_info, llm_arena_chatbot_right_info, llm_arena_input, llm_arena_answer, llm_arena_chatbot_left, llm_arena_chatbot_right]).then(
                llm_arena_func_chat, [llm_arena_chatbot_left_info, llm_arena_input, llm_arena_output], [llm_arena_chatbot_left, llm_arena_output], show_progress=True).then(
                llm_arena_func_chat, [llm_arena_chatbot_right_info, llm_arena_input, llm_arena_output], [llm_arena_chatbot_right, llm_arena_output], show_progress=True).then(
                common_clear, llm_arena_state, llm_arena_state # 每次都清空这个radio
                )
            
            llm_arena_both_button.click(lambda: "tie", [], llm_arena_state)
            llm_arena_left_button.click(lambda: "left_better", [], llm_arena_state)
            llm_arena_right_button.click(lambda: "right_better", [], llm_arena_state)
            llm_arena_bad_button.click(lambda: "bad", [], llm_arena_state)

            with gr.Tab("排行榜"):
                with gr.Row():
                    llm_arena_update_button = gr.Button(value="Update")
                with gr.Row():
                    llm_arena_leadboard = gr.DataFrame(headers=["Model", "Win Rate"])
            llm_arena_update_button.click(llm_arena_instance.get_win_rate, [], [llm_arena_leadboard])
                

            with gr.Tab("指令"):
                with gr.Row():
                    llm_arena_prompt_question = gr.Textbox(label="问题")
                with gr.Row():
                    llm_arena_prompt_task_type = gr.Textbox(label="任务类型")
                with gr.Row():
                    llm_arena_prompt_answer = gr.Textbox(label="参考回答")
                llm_arena_prompt_submit_button = gr.Button(value="Submit")

            def llm_arena_func_submit_prompt(question, task_type, answer):
                if len(question.strip())==0:
                    gr.Warning(f"请填写问题")
                    return question, task_type, answer
                if len(task_type.strip())==0:
                    gr.Warning(f"请填写任务类型")
                    return question, task_type, answer
                if len(answer.strip())==0:
                    gr.Warning(f"请填写参考回答")
                    return question, task_type, answer
                llm_arena_instance.save_question(question=question, task_type=task_type, answer=answer)
                question, task_type, answer = "", "", ""
                gr.Info(f"提交成功，感谢您的参与！")
                return question, task_type, answer
            
            llm_arena_prompt_submit_button.click(llm_arena_func_submit_prompt, [llm_arena_prompt_question, llm_arena_prompt_task_type, llm_arena_prompt_answer], 
                                                 [llm_arena_prompt_question, llm_arena_prompt_task_type, llm_arena_prompt_answer])

            with gr.Tab("说明"):
                gr.Markdown(llm_arena_instance.get_intro)
        
        with gr.Tab("LLM Data"):
            with gr.Row():
                with gr.Column(scale=1, variant='compact'):
                    with gr.Row(variant='default'):
                        with gr.Tab("Bot"):
                            with gr.Group():
                                llm_data_name_bot = gr.Dropdown(choices=llm_data_instance.get_preset_models(), label="Bot-Name", allow_custom_value=True, show_label=True)
                                llm_data_url_bot = gr.Textbox(label="Bot-URL")
                                
                            with gr.Group():
                                llm_data_instruction_name = gr.Dropdown(choices=llm_data_instance.get_preset_prompts(), label="Instruction-Name", allow_custom_value=True, show_label=True)
                                llm_data_update_instruct_button = gr.Button(value="Update Instruction", interactive=True)
                            with gr.Group():
                                llm_data_get_default_parameters_button = gr.Button(value="Get Default Parameters", interactive=True)
                                llm_data_parameters = gr.Textbox(label="Generation Parameters", value="{}")
                            with gr.Group():
                                gr.Markdown("""## 使用说明
                                1. Bot-Name: 选择要用于处理数据的大模型接口。
                                2. Instruction: 选择处理数据的Instruction.
                                3. Load Instruction: 加载Instruction.
                                3. Edit Instruction (Option): 编辑Instruction的system和few shot.
                                4. Chat: 输入待处理数据与大模型交互。""")
                with gr.Column(scale=4):
                    with gr.Row(variant='default'):
                        with gr.Group():
                            with gr.Tab("Instruction"):
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        llm_data_instruction_system = gr.Textbox(label="Instruction-System", lines=30)
                                    with gr.Column(scale=2):
                                        llm_data_instruction_fewshot = gr.Textbox(label="Instruction-Fewshot(chatml格式)", lines=30)
                                with gr.Row():
                                    llm_data_load_instruction_content_button = gr.Button(value="Load Instruction")
                                    llm_data_save_instruction_content_button = gr.Button(value="Save Instruction")
                    
                            with gr.Tab("Chat"):
                                with gr.Row():
                                    llm_data_chatbot = gr.Chatbot(label="LLM Bot", show_share_button=True, show_copy_button=True, show_copy_all_button=True)
                            
                                with gr.Row():
                                    llm_data_msg = gr.Textbox(label="Input", lines=4)
                                with gr.Row():
                                    llm_data_output = gr.Textbox(label="Output", interactive=False)
                                with gr.Row():
                                    llm_data_submit_button = gr.Button(value="🚀 Submit (发送)")
                                    llm_data_regenerate_button = gr.Button(value="🚀 Renenerate (重试)")
                                    llm_data_clear_button = gr.ClearButton(
                                            [llm_data_msg, llm_data_chatbot, llm_data_output, llm_data_parameters], 
                                            value="🧹 Clear (清除历史)")  


            llm_data_name_bot.change(llm_data_instance.select_preset_model, [llm_data_name_bot, llm_data_output], [llm_data_url_bot, llm_data_output])
            llm_data_instruction_name.change(llm_data_instance.load_instruction_content, [llm_data_instruction_name, llm_data_output], [llm_data_instruction_system, llm_data_instruction_fewshot, llm_data_output])
            llm_data_update_instruct_button.click(llm_data_instance.update_instruction, [], llm_data_instruction_name)
            llm_data_get_default_parameters_button.click(LLMData.get_default_parameters, [llm_data_url_bot, llm_data_output], [llm_data_parameters, llm_data_output])
            llm_data_load_instruction_content_button.click(llm_data_instance.load_instruction_content, [llm_data_instruction_name, llm_data_output], [llm_data_instruction_system, llm_data_instruction_fewshot, llm_data_output])
            llm_data_save_instruction_content_button.click(llm_data_instance.save_instruction_content, [llm_data_instruction_name, llm_data_instruction_system, llm_data_instruction_fewshot, llm_data_output], [llm_data_output])
            llm_data_submit_button.click(common_clear, llm_data_output, llm_data_output).then(
                LLMData.chat_stream, [llm_data_url_bot, llm_data_instruction_system, llm_data_instruction_fewshot, llm_data_msg, llm_data_parameters, llm_data_output], [llm_data_chatbot, llm_data_output], show_progress=True)
            llm_data_regenerate_button.click(common_clear, llm_data_output, llm_data_output).then(
                LLMData.chat_stream, [llm_data_url_bot, llm_data_instruction_system, llm_data_instruction_fewshot, llm_data_msg, llm_data_parameters, llm_data_output], [llm_data_chatbot, llm_data_output], show_progress=True)
            llm_data_msg.submit(common_clear, llm_data_output, llm_data_output).then(
                LLMData.chat_stream, [llm_data_url_bot, llm_data_instruction_system, llm_data_instruction_fewshot, llm_data_msg, llm_data_parameters, llm_data_output], [llm_data_chatbot, llm_data_output], show_progress=True)

        with gr.Tab("Debug Single"):
            with gr.Row():
                with gr.Column(scale=1, variant='compact'):
                    with gr.Row(variant='default'):
                        with gr.Tab("Bot"):
                            with gr.Group():
                                single_tab_name_bot = gr.Dropdown(choices=llm_debug_instance.get_preset_models(), label="Bot-Name", allow_custom_value=True, show_label=True)
                                single_tab_url_bot = gr.Textbox(label="Bot-URL")
                            with gr.Group():
                                single_tab_tts_voice_name = gr.Dropdown(
                                        choices=llm_debug_instance.get_tts_voice_names(), 
                                        value=llm_debug_instance.get_tts_voice_names()[0],
                                        label="TTS Voice Name",
                                        info="*shu*(例如meishulin)是基于大模型的TTS",
                                        allow_custom_value=True
                                    )
                                single_tab_tts = gr.Audio(label="Bot TTS Play", format="wav", scale=1, autoplay=False)
                            
                            with gr.Group():
                                single_tab_get_default_parameters_button = gr.Button(value="Get Default Parameters", interactive=True)
                                single_tab_parameters = gr.Textbox(label="Generation Parameters", value="{}")
                    # with gr.Row(variant='panel'):    
                    #     single_tab_model_list = gr.Textbox(label="Model List", info="You can find the model list from this address.", value="https://km.sankuai.com/collabpage/2200408702")
                
                with gr.Column(scale=4):
                    with gr.Row():
                        with gr.Column(scale=4):
                            single_tab_chatbot = gr.Chatbot(label="LLM Bot", height=500, show_copy_button=True, layout='panel', container=False)
                    with gr.Group():
                        with gr.Accordion("System", open=False):
                            with gr.Row():
                                with gr.Column(scale=4):
                                    single_tab_system = gr.Textbox(label="System")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                single_tab_msg = gr.Textbox(label="Input")
                            with gr.Column(scale=1):
                                single_tab_asr_input = gr.Audio(sources=["microphone"], label="Bot Asr", waveform_options=gr.WaveformOptions(waveform_color="#01C6FF",
                                waveform_progress_color="#0066B4",skip_length=2,show_controls=False) )  
                        single_tab_output = gr.Textbox(label="Output", interactive=False, max_lines=2)
                    with gr.Row():
                        single_tab_submit_button = gr.Button(value="🚀 Submit (发送)")
                        single_tab_regenerate_button = gr.Button(value="🚀 Renenerate (重试)")
                        single_tab_clear_button = gr.ClearButton(
                            [single_tab_msg, single_tab_chatbot, single_tab_tts, single_tab_output, single_tab_parameters], 
                            value="🧹 Clear (清除历史)")  
            single_tab_name_bot.select(fn=llm_debug_instance.select_model_bot,
                                       inputs=[single_tab_name_bot, single_tab_url_bot, single_tab_output], 
                                       outputs=[single_tab_name_bot, single_tab_url_bot, single_tab_output])   
            
            single_tab_get_default_parameters_button.click(fn=llm_debug_instance.get_default_parameters, 
                                                    inputs=[single_tab_url_bot, single_tab_output],
                                                    outputs=[single_tab_parameters, single_tab_output])
            single_tab_submit_button.click(common_clear, [single_tab_output], [single_tab_output]).then(
                        LLMDebug.response_wrapper_llm, 
                        [single_tab_url_bot, single_tab_system, single_tab_msg, single_tab_chatbot, single_tab_parameters, single_tab_output], 
                        [single_tab_chatbot, single_tab_output], show_progress=True).then(
                        common_clear, [single_tab_msg], [single_tab_msg]).then(
                            llm_debug_instance.response_wrapper_tts,
                            [single_tab_tts_voice_name, single_tab_chatbot, single_tab_output], 
                            [single_tab_tts, single_tab_output], show_progress=True
                        )
            single_tab_regenerate_button.click(common_clear, [single_tab_output], [single_tab_output]).then(
                        LLMDebug.regenerate_llm, 
                        [single_tab_url_bot, single_tab_system, single_tab_msg, single_tab_chatbot, single_tab_parameters, single_tab_output], 
                        [single_tab_chatbot, single_tab_output], show_progress=True).then(
                        common_clear, [single_tab_msg], [single_tab_msg]).then(
                            llm_debug_instance.response_wrapper_tts,
                            [single_tab_tts_voice_name, single_tab_chatbot, single_tab_output], 
                            [single_tab_tts, single_tab_output], show_progress=True
                        )
            single_tab_msg.submit(common_clear, [single_tab_output], [single_tab_output]).then(
                        LLMDebug.response_wrapper_llm, 
                        [single_tab_url_bot, single_tab_system, single_tab_msg, single_tab_chatbot, single_tab_parameters, single_tab_output], 
                        [single_tab_chatbot, single_tab_output], show_progress=True).then(
                        common_clear, [single_tab_msg], [single_tab_msg]).then(
                            llm_debug_instance.response_wrapper_tts,
                            [single_tab_tts_voice_name, single_tab_chatbot, single_tab_output], 
                            [single_tab_tts, single_tab_output], show_progress=True
                        )
            single_tab_asr_input.stop_recording(llm_debug_instance.response_wrapper_asr, 
                                                [single_tab_msg, single_tab_asr_input, single_tab_output], 
                                                [single_tab_msg, single_tab_output], 
                                                show_progress=True).then(common_clear, [single_tab_asr_input], [single_tab_asr_input])

        with gr.Tab("Debug SidebySide"):
            with gr.Row():
                with gr.Column(scale=1, variant='compact'):
                    with gr.Row(variant='default'):
                        with gr.Tab("Bot1"):
                            with gr.Group():
                                name_bot1 = gr.Dropdown(choices=llm_debug_instance.get_preset_models(), label="Bot1-Name", allow_custom_value=True)
                                url_bot1 = gr.Textbox(label="Bot1-URL")
                            with gr.Group():
                                tts_voice_name_left = gr.Dropdown(
                                        choices=llm_debug_instance.get_tts_voice_names(),
                                        value=llm_debug_instance.get_tts_voice_names()[0], 
                                        label="Bot1 TTS Voice Name",
                                        info="*shu*(meishujian)是基于大模型的TTS",
                                        allow_custom_value=True
                                    )
                                tts_left = gr.Audio(label="Bot1 TTS Play", format="wav", autoplay=False)
                            
                            with gr.Group():
                                get_default_parameters_button_left = gr.Button(value="Get Default Parameters", interactive=True)
                                parameters_left = gr.Textbox(label="Generation Parameters", value="{}")
                            
                        with gr.Tab("Bot2"):
                            with gr.Group():
                                name_bot2 = gr.Dropdown(choices=llm_debug_instance.get_preset_models(), label="Bot2-Name", allow_custom_value=True)
                                url_bot2 = gr.Textbox(label="Bot2-URL")
                            with gr.Group():
                                tts_voice_name_right = gr.Dropdown(
                                    choices=llm_debug_instance.get_tts_voice_names(), 
                                    value=llm_debug_instance.get_tts_voice_names()[0],
                                    label="Bot2 TTS Voice Name",
                                    info="*shu*(meishujian)是基于大模型的TTS",
                                    allow_custom_value=True
                                )
                                tts_right = gr.Audio(label="Bot2 TTS Play", format="wav", scale=1, autoplay=False)
                            
                            with gr.Group():
                                get_default_parameters_button_right = gr.Button(value="Get Default Parameters", interactive=True)
                                parameters_right = gr.Textbox(label="Generation Parameters", value="{}")
                        
                with gr.Column(scale=4):
                    with gr.Row():
                        with gr.Column(scale=2):
                            chatbot_left = gr.Chatbot(label="LLM Bot1", height=500, show_copy_button=True, layout='panel', container=False)
                            
                        with gr.Column(scale=2):
                            chatbot_right = gr.Chatbot(label="LLM Bot2", height=500, show_copy_button=True, layout='panel', container=False)
                            
                    with gr.Group():
                        with gr.Accordion("System", open=False):
                            with gr.Row():
                                with gr.Column(scale=4):
                                    system = gr.Textbox(label="System")
                        with gr.Row():
                            with gr.Column(scale=4):
                                msg = gr.Textbox(label="Input")
                        output = gr.Textbox(label="Output", interactive=False, max_lines=2)
                    with gr.Row():
                        submit_button = gr.Button(value="🚀 Submit (发送)")
                        regenerate_button = gr.Button(value="🚀 Regenerate (重试)")
                        clear_button = gr.ClearButton(
                            [msg, chatbot_left, chatbot_right, tts_left, tts_right, output, parameters_left, parameters_right], value="🧹 Clear (清除历史)")

            name_bot1.select(fn=llm_debug_instance.select_model_bot, inputs=[name_bot1, url_bot1, output], outputs=[name_bot1, url_bot1, output])
            name_bot2.select(fn=llm_debug_instance.select_model_bot, inputs=[name_bot2, url_bot2, output], outputs=[name_bot2, url_bot2, output])

            get_default_parameters_button_left.click(fn=llm_debug_instance.get_default_parameters, 
                                            inputs=[url_bot1, output],
                                            outputs=[parameters_left, output])
            get_default_parameters_button_right.click(fn=llm_debug_instance.get_default_parameters, 
                                            inputs=[url_bot2, output],
                                            outputs=[parameters_right, output])
            submit_button.click(common_clear, [output], [output]).then(LLMDebug.response_wrapper_llm, 
                        [url_bot1, system, msg, chatbot_left, parameters_left, output], 
                        [chatbot_left, output]).then(
                            LLMDebug.response_wrapper_llm, 
                        [url_bot2, system, msg, chatbot_right, parameters_right, output], 
                        [chatbot_right, output]).then(
                        common_clear, [msg], [msg]).then(
                            llm_debug_instance.response_wrapper_tts,
                        [tts_voice_name_left, chatbot_left, output],
                        [tts_left, output], show_progress=True).then(
                            llm_debug_instance.response_wrapper_tts,
                        [tts_voice_name_right, chatbot_right, output],
                        [tts_right, output], show_progress=True)
            regenerate_button.click(common_clear, [output], [output]).then(LLMDebug.regenerate_llm, 
                        [url_bot1, system, msg, chatbot_left, parameters_left, output], 
                        [chatbot_left, output]).then(
                            LLMDebug.regenerate_llm, 
                        [url_bot2, system, msg, chatbot_right, parameters_right, output], 
                        [chatbot_right, output]).then(
                        common_clear, [msg], [msg]).then(
                            llm_debug_instance.response_wrapper_tts,
                        [tts_voice_name_left, chatbot_left, output],
                        [tts_left, output], show_progress=True).then(
                            llm_debug_instance.response_wrapper_tts,
                        [tts_voice_name_right, chatbot_right, output],
                        [tts_right, output], show_progress=True)
            msg.submit(common_clear, [output], [output]).then(LLMDebug.response_wrapper_llm, 
                        [url_bot1, system, msg, chatbot_left, parameters_left, output], 
                        [chatbot_left, output]).then(
                            LLMDebug.response_wrapper_llm, 
                        [url_bot2, system, msg, chatbot_right, parameters_right, output], 
                        [chatbot_right, output]).then(
                        common_clear, [msg], [msg]).then(
                            llm_debug_instance.response_wrapper_tts,
                        [tts_voice_name_left, chatbot_left, output],
                        [tts_left, output], show_progress=True).then(
                            llm_debug_instance.response_wrapper_tts,
                        [tts_voice_name_right, chatbot_right, output],
                        [tts_right, output], show_progress=True)
        
        with gr.Tab("Vllm Deployment"):
            with gr.Group():
                with gr.Tab("Submit"):
                    # with gr.Group():
                        with gr.Row():
                            deploy_submit_user_name = gr.Textbox(label="user_name", scale=2)
                            deploy_submit_model_type = gr.Dropdown(label="model_type", choices=llm_deployment_instance.get_preset_models(), scale=2)
                            deploy_submit_model_path = gr.Textbox(label=f"model_path: (absolute path, start with {llm_deployment_instance.dolphin_root})", scale=4) 
                        deploy_submit_button = gr.Button(value="submit")
                with gr.Tab("Stop"):
                    # with gr.Group():
                        with gr.Row():
                            deploy_stop_user_name = gr.Textbox(label="user_name")
                            deploy_stop_url = gr.Textbox(label="url")
                            deploy_stop_pid = gr.Textbox(label="pid")
                        deploy_stop_button = gr.Button(value="stop")
            with gr.Group():
                deploy_update_status_button = gr.Button(value="Update Status")
                deploy_output = gr.JSON(label="Output")
            deploy_submit_button.click(fn=llm_deployment_instance.submit_deploy_model, 
                                       inputs=[deploy_submit_user_name, deploy_submit_model_type, deploy_submit_model_path],
                                       outputs=[deploy_output])
            deploy_stop_button.click(fn=llm_deployment_instance.stop_deploy_model,
                                     inputs=[deploy_stop_user_name, deploy_stop_url, deploy_stop_pid],
                                     outputs=[deploy_output], show_progress=True)
            deploy_update_status_button.click(fn=llm_deployment_instance.update_deploy_status,
                                              inputs=[],
                                              outputs=[deploy_output])

        with gr.Tab("Opencompass Evaluation"):
            with gr.Group():
                with gr.Tab('Submit'):
                    # local model submit
                    # with gr.Group():
                        with gr.Row(variant='panel'):
                            eval_submit_user_name = gr.Textbox(label="user_name", scale=2)
                            eval_submit_url = gr.Textbox(label="url", scale=2)
                            eval_submit_mode = gr.Dropdown(choices=llm_evaluation_instance.get_preset_modes(), value=llm_evaluation_instance.get_preset_modes()[-1],
                                                            label="mode", scale=1)
                            eval_submit_model = gr.Dropdown(choices=llm_evaluation_instance.get_preset_models(), value=llm_evaluation_instance.get_preset_models()[0],
                                                            label="model_type", scale=1)
                        with gr.Row(variant='panel'):
                            with gr.Group():
                                with gr.Tab("Objective Datasets"):
                                    eval_submit_objective_datasets = gr.CheckboxGroup(choices=llm_evaluation_instance.get_preset_datasets("objective"), info="常用的客观评估数据集", show_label=False)
                                with gr.Tab("Subjective Datasets"):
                                    eval_submit_subjective_datasets = gr.CheckboxGroup(choices=llm_evaluation_instance.get_preset_datasets("subjective"), info="主观评测数据集使用gpt4评估，请合理使用", show_label=False)
                    
                        eval_submit_button = gr.Button(value="Submit")
                
                with gr.Tab('Stop'):
                    # with gr.Group():
                        with gr.Row():
                            eval_delete_id = gr.Textbox(label="id")
                            eval_delete_user_name = gr.Textbox(label="user_name")
                            eval_delete_url = gr.Textbox(label="url")
                        eval_stop_button = gr.Button(value="Stop")
                        
            with gr.Group():
                eval_update_button = gr.Button(value="Update Status")
                eval_task_list = gr.JSON(label="Output")
                    

            eval_update_button.click(llm_evaluation_instance.update_eval_status, 
                                [], [eval_task_list])
            eval_submit_button.click(llm_evaluation_instance.start_eval_task, 
                                [eval_submit_user_name, eval_submit_mode, eval_submit_model, eval_submit_url, eval_submit_objective_datasets, eval_submit_subjective_datasets], 
                                [eval_task_list])
            eval_stop_button.click(llm_evaluation_instance.stop_eval_task, 
                                [eval_delete_id, eval_delete_user_name, eval_delete_url], 
                                [eval_task_list], show_progress=True)
            
        with gr.Tab("BM Evaluation"):
            with gr.Group():
                with gr.Tab('Submit'):
                    # local model submit
                    # with gr.Group():
                        with gr.Row(variant='panel'):
                            bm_eval_submit_user_name = gr.Textbox(label="user_name", scale=2)
                            bm_eval_submit_url = gr.Textbox(label="url", scale=2)
                            bm_eval_submit_model = gr.Dropdown(choices=bm_llm_evaluation_instance.get_preset_models(), value=bm_llm_evaluation_instance.get_preset_models()[0],
                                                            label="model_type", scale=1)
                        with gr.Row(variant='panel'):
                            bm_eval_submit_objective_datasets = gr.CheckboxGroup(choices=bm_llm_evaluation_instance.get_preset_datasets(), info="评估数据集", show_label=False)
                           
                        bm_eval_submit_button = gr.Button(value="Submit")
                
                with gr.Tab('Stop'):
                    # with gr.Group():
                        with gr.Row():
                            bm_eval_delete_id = gr.Textbox(label="id")
                            bm_eval_delete_user_name = gr.Textbox(label="user_name")
                            bm_eval_delete_url = gr.Textbox(label="url")
                        bm_eval_stop_button = gr.Button(value="Stop")
                        
            with gr.Group():
                bm_eval_update_button = gr.Button(value="Update Status")
                bm_eval_task_list = gr.JSON(label="Output")
                    

            bm_eval_update_button.click(bm_llm_evaluation_instance.update_eval_status, 
                                [], [bm_eval_task_list])
            bm_eval_submit_button.click(bm_llm_evaluation_instance.start_eval_task, 
                                [bm_eval_submit_user_name, bm_eval_submit_url, bm_eval_submit_objective_datasets, bm_eval_submit_model], 
                                [bm_eval_task_list])
            bm_eval_stop_button.click(bm_llm_evaluation_instance.stop_eval_task, 
                                [bm_eval_delete_id, bm_eval_delete_user_name, bm_eval_delete_url], 
                                [bm_eval_task_list], show_progress=True)

        with gr.Tab("VLM"):
            with gr.Row():
                with gr.Column(scale=1, variant='compact'): 
                    with gr.Row(variant='default'):
                        with gr.Tab("Bot"):
                            with gr.Group():
                                vlm_model_type = gr.Dropdown(label="model_type", choices=vlm_server_instance.get_preset_models())
                                vlm_model_url = gr.Textbox(label="url")
                            with gr.Group():
                                vlm_get_default_parameters_button = gr.Button(value="Get Default Parameters", interactive=True)
                                vlm_parameters = gr.Textbox(label="Generation Parameters", value="{}")
                            with gr.Group():
                                vlm_image = gr.Image(type='pil', label="Image as the first query.")
                            
                with gr.Column(scale=4):
                    vlm_chatbot = gr.Chatbot(label='VLM Bot') # , elem_classes="control-height", height=500)
                    with gr.Group():
                        vlm_system = gr.Textbox(label="system")
                        vlm_query = gr.Textbox(lines=1, label='Input', value="")
                        vlm_output = gr.Textbox(label='Output', value="", interactive=False)
                    vlm_task_history = gr.State([])
                    
                    with gr.Row():
                        vlm_submit_btn = gr.Button("🚀 Submit (发送)")
                        vlm_regenerate_btn = gr.Button("Regenerate (重试)")
                        vlm_addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])
                        vlm_clear_button = gr.ClearButton(value="🧹 Clear (清除历史)", components=[vlm_task_history, vlm_chatbot, vlm_output])
                        # regen_btn = gr.Button("🤔️ Regenerate (重试)")
                        

            vlm_model_type.select(vlm_server_instance.select_model_bot,
                                  [vlm_model_type, vlm_model_url, vlm_output],
                                  [vlm_model_type, vlm_model_url, vlm_output])
            vlm_submit_btn.click(common_clear, [vlm_output], [vlm_output]
                             ).then(vlm_server_instance.response_wrapper_vlm, 
                            [vlm_model_url, vlm_image, vlm_query, vlm_system, vlm_chatbot, vlm_task_history, vlm_output], 
                            [vlm_chatbot, vlm_task_history, vlm_output], show_progress=True
                            ).then(common_clear, [vlm_query], [vlm_query])
            vlm_regenerate_btn.click(vlm_server_instance.regenerate, 
                            [vlm_model_url, vlm_image, vlm_system, vlm_chatbot, vlm_task_history, vlm_output], 
                            [vlm_chatbot, vlm_task_history, vlm_output], show_progress=True
                            )
            vlm_get_default_parameters_button.click(fn=vlm_server_instance.get_default_parameters, 
                                                    inputs=[vlm_model_url, vlm_output],
                                                    outputs=[vlm_parameters, vlm_output])
            
            vlm_query.submit(common_clear, [vlm_output], [vlm_output]
                             ).then(vlm_server_instance.response_wrapper_vlm, 
                            [vlm_model_url, vlm_image, vlm_query, vlm_system, vlm_chatbot, vlm_task_history, vlm_output], 
                            [vlm_chatbot, vlm_task_history, vlm_output], show_progress=True
                            ).then(common_clear, [vlm_query], [vlm_query])
            vlm_addfile_btn.upload(vlm_server_instance.add_file, [vlm_chatbot, vlm_task_history, vlm_addfile_btn, vlm_output], [vlm_chatbot, vlm_task_history, vlm_output], show_progress=True)
        with gr.Tab("Voice annotation"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        # 添加一个状态组件来存储当前索引
                        voice_anno_current_index = gr.State(value=0)
                        
                        voice_anno_progress = gr.Textbox(label="进度", value="0/0", interactive=False)
                        voice_anno_prev_btn = gr.Button(value="上一个")
                        voice_anno_next_btn = gr.Button(value="下一个")
                    
                    with gr.Group():
                        # 添加录音状态显示
                        voice_anno_status = gr.Textbox(label="录音状态", value="未录制", interactive=False)
                        # 添加播放录音的组件
                        voice_anno_playback = gr.Audio(label="播放当前录音", interactive=False, visible=True)
                        voice_anno_record = gr.Audio(sources=["microphone"], label="录音", type="numpy")
                        voice_anno_save_btn = gr.Button(value="保存录音")
                        voice_anno_output = gr.Textbox(label="输出信息", interactive=False)
                
                    # 显示缺失录音的部分
                    with gr.Group():
                        voice_anno_missing_info = gr.Textbox(label="缺失录音信息", interactive=False)
                        # 将下拉框替换为数字输入框
                        voice_anno_jump_input = gr.Number(label="跳转到问答对编号", interactive=True, minimum=1, step=1)
                        voice_anno_jump_btn = gr.Button(value="跳转")
            
                with gr.Column(scale=2):
                    with gr.Group():
                        voice_anno_question = gr.Textbox(label="问题", lines=3, interactive=False)
                        voice_anno_answer = gr.Textbox(label="答案", lines=10, interactive=False)
            
            # 初始化显示第一个问题-答案对并检查缺失录音
            def init_voice_anno():
                question, answer, progress, current_index = voice_annotation_instance.get_current_qa()
                missing_info = voice_annotation_instance.get_missing_recordings_info()
                
                # 获取当前录音状态和路径
                has_recording, recording_path, status = voice_annotation_instance.get_current_recording_status()
                
                return question, answer, progress, missing_info, status, recording_path, current_index
            
            # 在所有标签页定义之后，但仍在 gr.Blocks 上下文内部添加 load 事件
            demo.load(
                fn=init_voice_anno,
                inputs=[],
                outputs=[voice_anno_question, voice_anno_answer, voice_anno_progress, 
                        voice_anno_missing_info, voice_anno_status, voice_anno_playback, voice_anno_current_index]  # 移除下拉框输出
            )
            
            # 更新函数：在切换问答对后更新缺失录音信息和播放状态
            def update_qa_and_missing_and_playback(current_index, next_or_prev="next"):
                # 根据参数决定调用哪个方法，并传递当前索引
                if next_or_prev == "next":
                    results = voice_annotation_instance.next_qa(current_index)
                else:  # prev
                    results = voice_annotation_instance.prev_qa(current_index)
                
                missing_info = voice_annotation_instance.get_missing_recordings_info()
                
                # 获取当前录音状态和路径
                has_recording, recording_path, status = voice_annotation_instance.get_current_recording_status()
                
                return results[0], results[1], results[2], missing_info, status, recording_path, results[3]  # 返回新的索引
            
            # 切换到下一个问题-答案对
            voice_anno_next_btn.click(
                fn=common_clear,
                inputs=[voice_anno_output],
                outputs=[voice_anno_output]
            ).then(
                fn=update_qa_and_missing_and_playback,
                inputs=[voice_anno_current_index],  # 传递当前索引
                outputs=[voice_anno_question, voice_anno_answer, voice_anno_progress, 
                        voice_anno_missing_info, voice_anno_status, voice_anno_playback, voice_anno_current_index]  # 更新索引
            )
            
            # 切换到上一个问题-答案对
            voice_anno_prev_btn.click(
                fn=common_clear,
                inputs=[voice_anno_output],
                outputs=[voice_anno_output]
            ).then(
                fn=lambda current_index: update_qa_and_missing_and_playback(current_index, "prev"),
                inputs=[voice_anno_current_index],  # 传递当前索引
                outputs=[voice_anno_question, voice_anno_answer, voice_anno_progress, 
                        voice_anno_missing_info, voice_anno_status, voice_anno_playback, voice_anno_current_index]  # 更新索引
            )
            
            # 跳转到选中的问答对
            def jump_to_selected_with_playback(selected_number, output_msg):
                if selected_number is None or selected_number < 1:
                    output_msg = append_output_msg(output_msg, "请输入有效的问答对编号")
                    return voice_annotation_instance.get_current_qa()[0], voice_annotation_instance.get_current_qa()[1], voice_annotation_instance.get_current_qa()[2], output_msg, "未录制", None, voice_annotation_instance.get_current_qa()[3]
                
                try:
                    # 直接使用输入的数字，减1转换为索引（因为UI中从1开始计数）
                    index = int(selected_number) - 1
                    question, answer, progress, current_index = voice_annotation_instance.jump_to_index(index)
                    output_msg = append_output_msg(output_msg, f"已跳转到问答对 #{index+1}")
                    
                    # 获取当前录音状态和路径
                    has_recording, recording_path, status = voice_annotation_instance.get_current_recording_status()
                    
                    return question, answer, progress, output_msg, status, recording_path, current_index
                except Exception as e:
                    output_msg = append_output_msg(output_msg, f"跳转失败: {str(e)}")
                    return voice_annotation_instance.get_current_qa()[0], voice_annotation_instance.get_current_qa()[1], voice_annotation_instance.get_current_qa()[2], output_msg, "未录制", None, voice_annotation_instance.get_current_qa()[3]
            
            voice_anno_jump_btn.click(
                fn=jump_to_selected_with_playback,
                inputs=[voice_anno_jump_input, voice_anno_output],
                outputs=[voice_anno_question, voice_anno_answer, voice_anno_progress, voice_anno_output, 
                        voice_anno_status, voice_anno_playback, voice_anno_current_index]  # 更新索引
            ).then(
                fn=voice_annotation_instance.get_missing_recordings_info,
                inputs=[],
                outputs=[voice_anno_missing_info]
            )
            
            # 保存录音后更新缺失录音信息和播放状态
            def save_audio_and_update_playback(audio, output_msg, current_index):
                # 传递当前索引到save_audio方法
                filepath, output_msg = voice_annotation_instance.save_audio(audio, output_msg, current_index)
                
                # 获取当前录音状态和路径
                has_recording, recording_path, status = voice_annotation_instance.get_current_recording_status()
                
                return filepath, output_msg, status, recording_path
            
            voice_anno_save_btn.click(
                fn=save_audio_and_update_playback,
                inputs=[voice_anno_record, voice_anno_output, voice_anno_current_index],
                outputs=[voice_anno_record, voice_anno_output, voice_anno_status, voice_anno_playback]
            ).then(
                fn=common_clear,
                inputs=[voice_anno_record],
                outputs=[voice_anno_record]
            ).then(
                fn=voice_annotation_instance.get_missing_recordings_info,
                inputs=[],
                outputs=[voice_anno_missing_info]
            )

        # 在Gradio界面中添加Voice Rank标签页
        with gr.Tab("Voice Rank"):
            with gr.Row():
                # 添加会话状态来存储当前组索引
                voice_rank_current_group = gr.State(value=0)
                
                with gr.Column(scale=1):
                    # 将缺失排名信息放在更明显的位置
                    with gr.Group(visible=True):  # 确保组件可见
                        voice_rank_missing_info = gr.Textbox(label="缺失排名信息", interactive=False, 
                                                            value="正在检查缺失排名...", lines=2)
                        voice_rank_jump_input = gr.Number(label="输入要跳转的组编号", interactive=True, minimum=1, step=1)
                        voice_rank_jump_btn = gr.Button(value="跳转到输入的组", size="sm", variant="primary")
                    
                    with gr.Group():
                        voice_rank_progress = gr.Textbox(label="进度", value="0/0", interactive=False)
                        with gr.Row():
                            voice_rank_prev_btn = gr.Button(value="上一组", size="sm")
                            voice_rank_next_btn = gr.Button(value="下一组", size="sm")
                    
                    with gr.Group():
                        voice_rank_save_btn = gr.Button(value="保存排名", size="lg", variant="primary")
                        voice_rank_output = gr.Textbox(label="输出信息", interactive=False)
                
                with gr.Column(scale=2):
                    with gr.Group():
                        # 显示问题
                        voice_rank_question = gr.Textbox(label="问题", lines=2, interactive=False)
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice_rank_sentence1 = gr.Textbox(label="回答1", lines=4, interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_audio1 = gr.Audio(label="音频", type="filepath", interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_rank1 = gr.Radio(
                                    label="口语化排名", 
                                    choices=["1 (最口语化)", "2", "3（最不口语化）"],
                                    value=None
                                )
                        # 添加备注输入框
                        voice_rank_note1 = gr.Textbox(label="备注1", lines=2, placeholder="请输入对回答1的备注")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice_rank_sentence2 = gr.Textbox(label="回答2", lines=4, interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_audio2 = gr.Audio(label="音频", type="filepath", interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_rank2 = gr.Radio(
                                    label="口语化排名", 
                                    choices=["1 (最口语化)", "2", "3（最不口语化）"],
                                    value=None
                                )
                        # 添加备注输入框
                        voice_rank_note2 = gr.Textbox(label="备注2", lines=2, placeholder="请输入对回答2的备注")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice_rank_sentence3 = gr.Textbox(label="回答3", lines=4, interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_audio3 = gr.Audio(label="音频", type="filepath", interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_rank3 = gr.Radio(
                                    label="口语化排名", 
                                    choices=["1 (最口语化)", "2", "3（最不口语化）"],
                                    value=None
                                )
                        # 添加备注输入框
                        voice_rank_note3 = gr.Textbox(label="备注3", lines=2, placeholder="请输入对回答3的备注")

    # 定义处理排名值的函数
        def process_rank_value(rank):
            if rank is None:
                return 0
            # 从选项中提取数字
            if isinstance(rank, str) and rank.startswith("1"):
                return 1
            elif isinstance(rank, str) and rank.startswith("2"):
                return 2
            elif isinstance(rank, str) and rank.startswith("3"):
                return 3
            return 0
        
        # 更新函数：在切换组后更新缺失排名信息
        def update_group_and_missing(fn, *args):
            results = fn(*args)
            missing_info, _ = voice_rank_instance.get_missing_rankings(0)
            return results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], missing_info

        # 初始化显示第一组数据并检查缺失排名
        def init_voice_rank():
            results = voice_rank_instance.get_group(0)  # 始终从索引0开始
            missing_info, _ = voice_rank_instance.get_missing_rankings(0)  # 传入当前索引0
            return (0, results[0], results[1], results[2], results[3], results[4], results[5], 
                    results[6], results[7], results[8], results[9], results[10], results[11], 
                    results[12], results[13], missing_info)

        # 在所有标签页定义之后，但仍在 gr.Blocks 上下文内部添加 load 事件
        demo.load(
            fn=init_voice_rank,
            inputs=[],
            outputs=[
                voice_rank_current_group,  # 添加当前组状态
                voice_rank_question,
                voice_rank_sentence1, voice_rank_audio1,
                voice_rank_sentence2, voice_rank_audio2,
                voice_rank_sentence3, voice_rank_audio3,
                voice_rank_progress,
                voice_rank_rank1, voice_rank_rank2, voice_rank_rank3,  # 添加排名输出
                voice_rank_note1, voice_rank_note2, voice_rank_note3,  # 添加备注输出
                voice_rank_missing_info
            ]
        )
        
        # 切换到下一组，修改为包含排名和备注
        voice_rank_next_btn.click(
            fn=common_clear,
            inputs=[voice_rank_output],
            outputs=[voice_rank_output]
        ).then(
            fn=voice_rank_instance.next_group,  # 传入当前索引
            inputs=[voice_rank_current_group],  # 使用会话中的当前索引
            outputs=[
                voice_rank_current_group,  # 更新当前索引
                voice_rank_question,
                voice_rank_sentence1, voice_rank_audio1,
                voice_rank_sentence2, voice_rank_audio2,
                voice_rank_sentence3, voice_rank_audio3,
                voice_rank_progress,
                voice_rank_rank1, voice_rank_rank2, voice_rank_rank3,  # 添加排名输出
                voice_rank_note1, voice_rank_note2, voice_rank_note3   # 添加备注输出
            ]
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_current_group],
            outputs=[voice_rank_missing_info]
        )
        
        # 切换到上一组，修改为包含排名和备注
        voice_rank_prev_btn.click(
            fn=common_clear,
            inputs=[voice_rank_output],
            outputs=[voice_rank_output]
        ).then(
            fn=voice_rank_instance.prev_group,
            inputs=[voice_rank_current_group],
            outputs=[
                voice_rank_current_group,
                voice_rank_question,
                voice_rank_sentence1, voice_rank_audio1,
                voice_rank_sentence2, voice_rank_audio2,
                voice_rank_sentence3, voice_rank_audio3,
                voice_rank_progress,
                voice_rank_rank1, voice_rank_rank2, voice_rank_rank3,  # 添加排名输出
                voice_rank_note1, voice_rank_note2, voice_rank_note3   # 添加备注输出
            ]
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_current_group],
            outputs=[voice_rank_missing_info]
        )
        
        # 跳转到输入的组
        def jump_to_input_group(group_number, current_index, output_msg):
            if not group_number or group_number < 1:
                output_msg = append_output_msg(output_msg, "请输入有效的组编号")
                return current_index, output_msg
            
            try:
                # 组编号从1开始，但索引从0开始
                index = int(group_number) - 1
                if 0 <= index < voice_rank_instance.total_groups:
                    output_msg = append_output_msg(output_msg, f"已跳转到组 #{index+1}")
                    return index, output_msg
                else:
                    output_msg = append_output_msg(output_msg, "组编号超出范围")
                    return current_index, output_msg
            except Exception as e:
                output_msg = append_output_msg(output_msg, f"跳转失败: {str(e)}")
                return current_index, output_msg

        voice_rank_jump_btn.click(
            fn=jump_to_input_group,
            inputs=[voice_rank_jump_input, voice_rank_current_group, voice_rank_output],
            outputs=[voice_rank_current_group, voice_rank_output]
        ).then(
            fn=voice_rank_instance.get_group,  # 使用 get_group 获取新索引的数据
            inputs=[voice_rank_current_group],
            outputs=[
                voice_rank_question,
                voice_rank_sentence1, voice_rank_audio1,
                voice_rank_sentence2, voice_rank_audio2,
                voice_rank_sentence3, voice_rank_audio3,
                voice_rank_progress,
                voice_rank_rank1, voice_rank_rank2, voice_rank_rank3,  # 添加排名输出
                voice_rank_note1, voice_rank_note2, voice_rank_note3   # 添加备注输出
            ]
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_current_group],
            outputs=[voice_rank_missing_info]
        )

        # 保存排名后更新缺失排名信息
        voice_rank_save_btn.click(
            fn=lambda r1, r2, r3, n1, n2, n3, current_index, msg: (
                voice_rank_instance.save_rank(
                    process_rank_value(r1),
                    process_rank_value(r2),
                    process_rank_value(r3),
                    0,  # 不再使用的排名位置
                    0,  # 不再使用的排名位置
                    n1, n2, n3,  # 添加备注参数
                    msg,
                    current_index  # 传入当前索引
                ),
                current_index  # 返回当前索引，不变
            ),
            inputs=[
                voice_rank_rank1, voice_rank_rank2, voice_rank_rank3, 
                voice_rank_note1, voice_rank_note2, voice_rank_note3,
                voice_rank_current_group,  # 添加当前索引作为输入
                voice_rank_output
            ],
            outputs=[voice_rank_output, voice_rank_current_group]  # 返回当前索引，确保不变
        ).then(
            fn=voice_rank_instance.get_missing_rankings,
            inputs=[voice_rank_current_group],
            outputs=[voice_rank_missing_info]
        )

        # 在Gradio界面中添加Voice Rank Multiple标签页
        with gr.Tab("Voice Rank Multiple"):
            with gr.Row():
                # 添加会话状态来存储当前组索引，类似于 Voice Rank 页面
                voice_rank_multiple_current_group = gr.State(value=0)
                
                with gr.Column(scale=1):
                    # 将缺失排名信息放在更明显的位置
                    with gr.Group(visible=True):  # 确保组件可见
                        voice_rank_multiple_missing_info = gr.Textbox(label="缺失排名信息", interactive=False, 
                                                value="正在检查缺失排名...", lines=2)
                        voice_rank_multiple_jump_input = gr.Number(label="输入要跳转的组编号", interactive=True, minimum=1, step=1)
                        voice_rank_multiple_jump_btn = gr.Button(value="跳转到输入的组", size="sm", variant="primary")
                    
                    with gr.Group():
                        voice_rank_multiple_progress = gr.Textbox(label="进度", value="0/0", interactive=False)
                        with gr.Row():
                            voice_rank_multiple_prev_btn = gr.Button(value="上一组", size="sm")
                            voice_rank_multiple_next_btn = gr.Button(value="下一组", size="sm")
                    
                    with gr.Group():
                        voice_rank_multiple_save_btn = gr.Button(value="保存排名", size="lg", variant="primary")
                        voice_rank_multiple_output = gr.Textbox(label="输出信息", interactive=False)
                
                with gr.Column(scale=2):
                    with gr.Group():
                        # 显示问题
                        voice_rank_multiple_question = gr.Textbox(label="问题", lines=2, interactive=False)
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice_rank_multiple_sentence1 = gr.Textbox(label="回答1", lines=4, interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_multiple_audio1 = gr.Audio(label="音频", type="filepath", interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_multiple_rank1 = gr.Radio(
                                    label="口语化排名", 
                                    choices=["1 (最口语化)", "2", "3（最不口语化）"],
                                    value=None
                                )
                        # 添加备注输入框
                        voice_rank_multiple_note1 = gr.Textbox(label="备注1", lines=2, placeholder="请输入对回答1的备注")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice_rank_multiple_sentence2 = gr.Textbox(label="回答2", lines=4, interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_multiple_audio2 = gr.Audio(label="音频", type="filepath", interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_multiple_rank2 = gr.Radio(
                                    label="口语化排名", 
                                    choices=["1 (最口语化)", "2", "3（最不口语化）"],
                                    value=None
                                )
                        # 添加备注输入框
                        voice_rank_multiple_note2 = gr.Textbox(label="备注2", lines=2, placeholder="请输入对回答2的备注")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice_rank_multiple_sentence3 = gr.Textbox(label="回答3", lines=4, interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_multiple_audio3 = gr.Audio(label="音频", type="filepath", interactive=False)
                            with gr.Column(scale=1):
                                voice_rank_multiple_rank3 = gr.Radio(
                                    label="口语化排名", 
                                    choices=["1 (最口语化)", "2", "3（最不口语化）"],
                                    value=None
                                )
                        # 添加备注输入框
                        voice_rank_multiple_note3 = gr.Textbox(label="备注3", lines=2, placeholder="请输入对回答3的备注")

        # 修改初始化函数，使用会话状态而不是实例变量
        def init_voice_rank_multiple():
            results = voice_rank_multiple_instance.get_group(0)  # 使用 get_group 而不是 get_current_group
            missing_info = voice_rank_multiple_instance.get_missing_rankings(0)[0]  # 只获取字符串部分
            return (0, results[0], results[1], results[2], results[3], results[4], results[5], 
                    results[6], results[7], results[8], results[9], results[10], results[11], 
                    results[12], results[13], missing_info)

        # 在所有标签页定义之后，但仍在 gr.Blocks 上下文内部添加 load 事件
        demo.load(
            fn=init_voice_rank_multiple,
            inputs=[],
            outputs=[
                voice_rank_multiple_current_group,  # 添加当前索引作为输出
                voice_rank_multiple_question,
                voice_rank_multiple_sentence1, voice_rank_multiple_audio1,
                voice_rank_multiple_sentence2, voice_rank_multiple_audio2,
                voice_rank_multiple_sentence3, voice_rank_multiple_audio3,
                voice_rank_multiple_progress,
                voice_rank_multiple_rank1, voice_rank_multiple_rank2, voice_rank_multiple_rank3,  # 添加排名输出
                voice_rank_multiple_note1, voice_rank_multiple_note2, voice_rank_multiple_note3,  # 添加备注输出
                voice_rank_multiple_missing_info
            ]
        )

        # 修改切换到下一组的函数，使用会话状态并包含排名和备注
        voice_rank_multiple_next_btn.click(
            fn=common_clear,
            inputs=[voice_rank_multiple_output],
            outputs=[voice_rank_multiple_output]
        ).then(
            fn=voice_rank_multiple_instance.next_group,  # 使用 next_group 函数并传入当前索引
            inputs=[voice_rank_multiple_current_group],  # 使用会话中的当前索引
            outputs=[
                voice_rank_multiple_current_group,  # 更新当前索引
                voice_rank_multiple_question,
                voice_rank_multiple_sentence1, voice_rank_multiple_audio1,
                voice_rank_multiple_sentence2, voice_rank_multiple_audio2,
                voice_rank_multiple_sentence3, voice_rank_multiple_audio3,
                voice_rank_multiple_progress
            ]
        ).then(
            # 获取当前组的排名和备注
            fn=lambda idx: voice_rank_multiple_instance.get_group(idx)[8:14],  # 获取排名和备注部分
            inputs=[voice_rank_multiple_current_group],
            outputs=[
                voice_rank_multiple_rank1, voice_rank_multiple_rank2, voice_rank_multiple_rank3,
                voice_rank_multiple_note1, voice_rank_multiple_note2, voice_rank_multiple_note3
            ]
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_multiple_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_multiple_current_group],
            outputs=[voice_rank_multiple_missing_info]
        )

        # 修改切换到上一组的函数，使用会话状态并包含排名和备注
        voice_rank_multiple_prev_btn.click(
            fn=common_clear,
            inputs=[voice_rank_multiple_output],
            outputs=[voice_rank_multiple_output]
        ).then(
            fn=voice_rank_multiple_instance.prev_group,  # 使用 prev_group 函数并传入当前索引
            inputs=[voice_rank_multiple_current_group],
            outputs=[
                voice_rank_multiple_current_group,
                voice_rank_multiple_question,
                voice_rank_multiple_sentence1, voice_rank_multiple_audio1,
                voice_rank_multiple_sentence2, voice_rank_multiple_audio2,
                voice_rank_multiple_sentence3, voice_rank_multiple_audio3,
                voice_rank_multiple_progress
            ]
        ).then(
            # 获取当前组的排名和备注
            fn=lambda idx: voice_rank_multiple_instance.get_group(idx)[8:14],  # 获取排名和备注部分
            inputs=[voice_rank_multiple_current_group],
            outputs=[
                voice_rank_multiple_rank1, voice_rank_multiple_rank2, voice_rank_multiple_rank3,
                voice_rank_multiple_note1, voice_rank_multiple_note2, voice_rank_multiple_note3
            ]
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_multiple_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_multiple_current_group],
            outputs=[voice_rank_multiple_missing_info]
        )

        # 修改跳转到输入的组的函数，使用会话状态并包含排名和备注
        def jump_to_input_group_multiple(group_number, current_index, output_msg):
            if not group_number or group_number < 1:
                output_msg = append_output_msg(output_msg, "请输入有效的组编号")
                return current_index, output_msg
            
            try:
                # 组编号从1开始，但索引从0开始
                index = int(group_number) - 1
                if 0 <= index < voice_rank_multiple_instance.total_groups:
                    output_msg = append_output_msg(output_msg, f"已跳转到组 #{index+1}")
                    return index, output_msg
                else:
                    output_msg = append_output_msg(output_msg, "组编号超出范围")
                    return current_index, output_msg
            except Exception as e:
                output_msg = append_output_msg(output_msg, f"跳转失败: {str(e)}")
                return current_index, output_msg

        # 修改点击事件处理
        voice_rank_multiple_jump_btn.click(
            fn=jump_to_input_group_multiple,
            inputs=[voice_rank_multiple_jump_input, voice_rank_multiple_current_group, voice_rank_multiple_output],
            outputs=[voice_rank_multiple_current_group, voice_rank_multiple_output]
        ).then(
            fn=voice_rank_multiple_instance.get_group,  # 使用 get_group 获取新索引的数据
            inputs=[voice_rank_multiple_current_group],
            outputs=[
                voice_rank_multiple_question,
                voice_rank_multiple_sentence1, voice_rank_multiple_audio1,
                voice_rank_multiple_sentence2, voice_rank_multiple_audio2,
                voice_rank_multiple_sentence3, voice_rank_multiple_audio3,
                voice_rank_multiple_progress,
                voice_rank_multiple_rank1, voice_rank_multiple_rank2, voice_rank_multiple_rank3,  # 添加排名输出
                voice_rank_multiple_note1, voice_rank_multiple_note2, voice_rank_multiple_note3   # 添加备注输出
            ]
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_multiple_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_multiple_current_group],
            outputs=[voice_rank_multiple_missing_info]
        )

        # 修改保存排名的函数，使用会话状态并包含排名和备注
        voice_rank_multiple_save_btn.click(
            fn=lambda r1, r2, r3, n1, n2, n3, current_index, msg: (
                voice_rank_multiple_instance.save_rank(
                    process_rank_value(r1),
                    process_rank_value(r2),
                    process_rank_value(r3),
                    0,  # 不再使用的排名位置
                    0,  # 不再使用的排名位置
                    n1, n2, n3,  # 添加备注参数
                    msg,
                    current_index  # 传入当前索引
                ),
                current_index  # 返回当前索引，不变
            ),
            inputs=[
                voice_rank_multiple_rank1, voice_rank_multiple_rank2, voice_rank_multiple_rank3, 
                voice_rank_multiple_note1, voice_rank_multiple_note2, voice_rank_multiple_note3,
                voice_rank_multiple_current_group,  # 添加当前索引作为输入
                voice_rank_multiple_output
            ],
            outputs=[voice_rank_multiple_output, voice_rank_multiple_current_group]  # 返回当前索引，确保不变
        ).then(
            # 只获取 missing_info 字符串，忽略 missing_groups 列表
            fn=lambda idx: voice_rank_multiple_instance.get_missing_rankings(idx)[0],
            inputs=[voice_rank_multiple_current_group],
            outputs=[voice_rank_multiple_missing_info]
        )

        with gr.Tab("About"):
            gr.Markdown(value=about_instance.read_intro)
    # 启动 Gradio 应用
    demo.launch(server_name=args.host, server_port=args.port, 
                ssl_verify=False, debug=True,
                max_file_size=50*1024*1024)

if __name__=="__main__":
    args = get_args()
    main(args)