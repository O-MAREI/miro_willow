#!/usr/bin/env python3
import numpy as np
import os
import rospy
from std_msgs.msg import Int16MultiArray, String, UInt32MultiArray, Float32MultiArray, UInt16MultiArray
from sensor_msgs.msg import JointState
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from pydub import AudioSegment
from io import BytesIO
import openai
import config
import logging
import uvicorn
from fastapi import FastAPI
import httpx
import argparse
import asyncio
from config_lovey import Settings
from lovey_logging import setup_logging
from message import WISMessage

#------------------------ Initialization -------------------------
rospy.init_node("process_audio")
openai.api_key = config.load_api_key()

color = UInt32MultiArray()
color.data = [0xFFFFFFFF] * 6
hey_miro = False
switch_modes = False  
talking = False

cos_cmd = Float32MultiArray()
cos_cmd.data = [0, 0.5, 0, 0, 0.3333, 0.3333]  # default for the eye (0.3333)
joint_cmd = JointState()
joint_cmd.position = [0, 0, 0, 0]
color_change = UInt32MultiArray()

tone = UInt16MultiArray()
tone.data = [0, 0, 0]
beep = False

state = "llm"
message = String()

model = "gpt-3.5-turbo"
message_history = [{"role": "system", "content": "You are a friendly robot assistant called MiRo."}]

client = ElevenLabs(api_key="fe78b860a763919b03861b0ef7f4ec0f")

topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
pub_gestures = rospy.Publisher(topic_base_name + "/gpt_speech/actions", String, queue_size=0)
pub_response = rospy.Publisher(topic_base_name + "/gpt_speech/prompt_response", Int16MultiArray, queue_size=0)
pub_stream = rospy.Publisher(topic_base_name + "/control/stream", Int16MultiArray, queue_size=0)
pub_kinematic = rospy.Publisher(topic_base_name + "/control/kinematic_joints", JointState, queue_size=0)
pub_cosmetic = rospy.Publisher(topic_base_name + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)
pub_illumination = rospy.Publisher(topic_base_name + "/control/illum", UInt32MultiArray, queue_size=0)
pub_tone = rospy.Publisher(topic_base_name + "/control/tone", UInt16MultiArray, queue_size=0)

# Define callback function
def command_callback(msg):
    global state, talking
    if msg.data == "switch_mode":
        state = "animal" if state == "llm" else "llm"
    else:
        chatGPT_process(msg.data)

rospy.Subscriber(topic_base_name + "/commands", String, command_callback)

#-------------------------------- Code ----------------------------------
class NamedBytesIO(BytesIO):
    def __init__(self, name, data=None):
        super().__init__(data if data is not None else b'')
        self.name = name

def set_move_kinematic(tilt=0, lift=0, yaw=0, pitch=0):
    joint_cmd.position = [tilt, lift, yaw, pitch]

async def chatGPT_process(message):
    global talking, data, d
    print("Processing prompt: ", message)
    user_prompt = {"role": "user", "content": message}
    message_history.append(user_prompt)

    if state == "llm":
        chat = openai.ChatCompletion.create(model=model, messages=message_history)
        reply = chat['choices'][0]['message']['content']
        print(f"Response from GPT: {reply}")

        set_move_kinematic(lift=0.59)
        pub_kinematic.publish(joint_cmd)

        text_to_wav(reply)
        
        rate = rospy.Rate(10)
        d = 0
        while not rospy.core.is_shutdown():
            if d < len(data):
                msg = Int16MultiArray(data=data[d:d + 1000])
                d += 1000
                pub_stream.publish(msg)
            else:
                talking = False
                break
            rate.sleep()
    elif state == "animal":
        movement = Movement(message)
        movement.main()

def text_to_wav(reply):
    global data
    audio = client.generate(
        text=reply,
        voice=Voice(voice_id="vGQNBgLaiM3EdZtxIiuY", settings=VoiceSettings(stability=0.5, similarity_boost=0.75))
    )
    audio_byte = b''.join(audio)
    audio_data = BytesIO(audio_byte)

    seg = AudioSegment.from_file(audio_data, format='mp3')
    seg = seg.set_frame_rate(8000)
    seg = seg.set_channels(1)

    wav_io = BytesIO()
    seg.export(wav_io, format='wav')
    wav_io.seek(0)

    wav_io.seek(44)
    dat = np.frombuffer(wav_io.read(), dtype=np.int16)
    wav_io.close()

    dat = dat.astype(float)
    sc = 32767.0 / np.max(np.abs(dat))
    dat *= sc
    dat = dat.astype(np.int16).tolist()
    
    data = dat
    talking = True

#------------------------------------- FastAPI ------------------------------------
logger = logging.getLogger(__name__)

settings = Settings()
model_submit_url = f'http://{settings.flsk_ip}:{settings.flsk2_port}/submit_to_model'

app = FastAPI(
    title='Rest API Microservice for Willow',
    description='A microservice for receiving transcriptions from WIS and sending them to the chat microservice',
    openapi_url="/docs/openapi.json",
    redoc_url="/docs",
)

async def submit_to_model(message: WISMessage) -> httpx.Response:
    '''Sends WIS transcription to the chat microservice'''
    async with httpx.AsyncClient() as client:
        return await client.post(model_submit_url, json=message.model_dump())

async def format_for_esp(text: str) -> str:
    '''Truncate the text if needed and return a formatted string'''
    max_ask_len = 36
    if max_ask_len < len(text):        
        split_index = text.rfind(' ', 0, max_ask_len)
        if split_index == -1:
            split_index = max_ask_len

        text = text[:split_index] + ', etc.'

    asyncio.create_task(chatGPT_process(text))
    return f'I heard: {text} Please wait. Transcription sent to MiRo'
    
@app.post('/rest4willow')
async def rest4willow(message: WISMessage) -> str:
    '''Receive a transcription from WIS and send it to the chat microservice.'''
    asyncio.create_task(submit_to_model(message))
    return await format_for_esp(message.text)

def main(args):
    defaults = {
        'log_dir': settings.log_dir,
    }
    setup_logging(args.logging_config, defaults)
    uvicorn.run(app, host=settings.flsk_ip, port=int(settings.flsk1_port))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rest API Microservice for Willow')
    parser.add_argument(
        '--logging_config',
        type=str,
        default='lovey_logging/configs/rest4willow_default.json',
        help='The path to the logging configuration JSON (default: %(default)s).')
    args = parser.parse_args()

    main(args)