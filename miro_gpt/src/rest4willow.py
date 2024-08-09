#!/usr/bin/env python3
# WARNING: Whole script may be merged into process chat microservice
import argparse
import asyncio
import logging

from fastapi import FastAPI
import httpx
import uvicorn

from config_lovey import Settings
from lovey_logging import setup_logging
from message import WISMessage

# Importing the ChatGPT processing function
import os
import sys
import subprocess


#process_audio_dir = os.path.abspath('/home/omar/mdk/catkin_ws/src/miro_willow/miro_gpt/src')
#sys.path.append(process_audio_dir)
#from process_audio import chatGPT_process
#from process_audio import chatGPT_process

#def install_package(package, version=None):
#    if version:
#        package = f"{package}=={version}"
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#install_package("opencv-python")

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

    ProcessAudio.chatGPT_process(text)
    return f'I heard: {text} Please wait. Sendingin to the MiRo'
    

@app.post('/rest4willow')
async def rest4willow(message: WISMessage) -> str:
    '''Receive a transcription from WIS and send it to the chat microservice.
    Returns a formatted string for the ESP to display while waiting for the
    model response.'''
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