from functools import cached_property
import json
import os
import pathlib
from typing import Any
import yaml

# My additions
from typing import Union

from langchain_community import vectorstores
from pydantic import computed_field, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import Conversation, Pipeline, pipeline

from conversation import ConversationWrapper, load_conversation


#https://stackoverflow.com/a/8249667
URL_OR_IP_REGEX = \
    r'^(http(s?):\/\/)?(((www\.)?+[a-zA-Z0-9\.\-\_]+(\.[a-zA-Z]{2,3})+)' \
    r'|(\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|' \
    r'2[0-4][0-9]|[01]?[0-9][0-9]?)\b))(\/[a-zA-Z0-9\_\-\s\.\/\?\%\#\&\=]*)?$'


class Settings(BaseSettings):
    '''Shared settings class for all microservices. Reads in environment
    variables from OS and .env file. OS environment variables take
    precedence'''
    # ===== Microservice IP and Port Settings ===== #
    flsk_ip: str = Field(
        '0.0.0.0',
        pattern = URL_OR_IP_REGEX,
        alias = '_lovey_services_ip',
        description = 'The IP address or URL of the microservices.',)
    flsk1_port: int = Field(
        19091,
        gt = 0,
        alias = '_lovey_flsk1_port',
        description = 'The port number for the rest4willow microservice.',)
    flsk2_port: int = Field(
        19092,
        gt = 0,
        alias = '_lovey_flsk2_port',
        description = 'The port number for the process_chat microservice.',)
    flsk3_port: int = Field(
        19093,
        gt = 0,
        alias = '_lovey_flsk3_port',
        description = 'The port number for the control devices microservice.',)
    
    # ===== Willow Application Server (WAS) Settings ===== #
    was_ip: str = Field(
        '0.0.0.0',
        pattern = URL_OR_IP_REGEX,
        alias = '_lovey_was_ip',
        description = 'The IP address or URL of the willow application server (WIS).',)
    was_port: int = Field(
        8502,
        gt = 0,
        alias = '_lovey_was_port',
        description = 'The port number for the willow application server (WIS).',)

    # ===== Willow Inference Server (WIS) Settings ===== #
    wis_ip: str = Field(
        'infer.tovera.io',
        pattern = URL_OR_IP_REGEX,
        alias = '_lovey_wis_ip',
        description = 'The IP address or URL of the willow inference server (WIS).',)
    wis_port: int = Field(
        443,
        gt = 0,
        alias = '_lovey_wis_port',
        description = 'The port number for the willow inference server (WIS).',)

    # ===== Misc. Settings ===== #
    log_dir: str = Field(
        'logs',
        alias = '_LOVEY_LOG_DIR',
        description = 'The directory where log files are stored.',)
    shut_secret: str = Field(
        'xxxTODOxxx',
        alias = '_LOVEY_SHUT_SECRET',
        description = '',) # TODO: Fill in description
    shut_dir: pathlib.Path = Field(
        '/tmp/lovey-shutdown',
        alias = '_LOVEY_SHUT_DIR',
        description = '',) # TODO: Fill in description
    shut_file: pathlib.Path = Field(
        'shutdown-request.txt',
        alias = '_LOVEY_SHUT_FILE',
        description = '',) # TODO: Fill in description
    
    model_config = SettingsConfigDict(env_file='.env')

    def model_post_init(self, __context: Any) -> None:           
        self.shut_file = os.path.join(self.shut_dir, self.shut_file)
        os.makedirs(self.shut_dir, exist_ok=True)

        conversations_dir = os.path.join(self.log_dir, 'conversations')
        os.makedirs(conversations_dir, exist_ok=True)

    @computed_field(description='The URL for the WAS to send notifications to the client.')
    @property
    def was_notify_url(self) -> str:
        return f'http://{self.was_ip}:{self.was_port}/api/client?action=notify'
    
    @computed_field(description='The base URL for the text-to-speech API.')
    @property
    def tts_base_url(self) -> str:
        return f'https://{self.wis_ip}:{self.wis_port}/api/tts?text='


class LanguageModelSettings(BaseSettings):
    '''Settings class for the language model related microservices'''
    # ===== Model Settings ===== #
    chat_model_name_or_path: Union[str, pathlib.Path] = Field(
        default = 'lmsys/vicuna-13b-v1.5',
        description = 'The model name or path for the chat model.',)
    chat_model_kwargs: dict = Field(
        default_factory = dict,
        description = 'Additional keyword arguments for the chat model.',)
    embedding_model_name_or_path: Union[str, pathlib.Path] = Field(
        default = 'sentence-transformers/all-mpnet-base-v2',
        description = 'The model name or path for the embedding model.',)
    embedding_model_kwargs: dict = Field(
        default_factory = dict,
        description = 'Additional keyword arguments for the embedding model.',)
    
    # ===== Chat Settings ===== #
    base_conversation_name: str = Field(
        default = 'default',
        description = 'The base conversation to be used from the base_conversations directory.',)
    conversation_vars: dict = Field(
        default_factory = dict,
        description = 'The variables to be formatted into the conversation strings.',)
    load_previous_conversation: bool = Field(
        default = False,
        description = 'Whether to load the previous conversation state to initialise the new model.',)
    
    # ===== Vectorising Settings ===== #
    vector_store_name: str = Field( # TODO: Add choices
        default = 'FAISS',
        description = 'The type of vector store to be used.',)
    vector_store_dir: pathlib.Path = Field(
        default = 'vector_store',
        description = 'The directory where the vector store is or will be located.',)
    
    # ===== Smart Home Settings ===== #
    smart_home_data_path: pathlib.Path = Field(
        default = 'smart-home-devices.json',
        description = 'The path to the smart home data JSON file.',)
    
    def model_post_init(self, __context: Any) -> None:
        self.base_conversation_name = self.base_conversation_name.rstrip('.json')

    @computed_field
    @cached_property
    def model(self) -> Pipeline:
        '''Lazy load the chat model.'''
        return pipeline(
            task='conversational',
            model=self.chat_model_name_or_path,
            model_kwargs=self.chat_model_kwargs,)
    
    @computed_field
    @cached_property
    def embedding_model(self) -> Any:
        '''Lazy load the embedding model.'''
        return NotImplementedError('Embedding model not yet implemented.')
    
    @computed_field
    @cached_property
    def smart_home_data(self) -> dict:
        '''Lazy load the smart home JSON file.'''
        with open(self.smart_home_data_path, 'r') as f:
            return json.load(f)
    
    @computed_field
    @cached_property
    def base_conversation(self) -> Conversation:
        '''Lazy Load the base conversation.'''
        convs_dir = os.path.join(os.path.dirname(__file__), 'conversation', 'templates')
        conv_path = os.path.join(convs_dir, f'{self.base_conversation_name}.json')
        return load_conversation(
            conv_path,
            smart_home_data=json.dumps(self.smart_home_data, indent='\t'),
            **self.conversation_vars,)  
    
    @computed_field
    @cached_property
    def loaded_conversation(self) -> Union[Conversation, None]:
        '''Load the previous conversation state if required.'''
        if self.load_previous_conversation:
            return NotImplementedError('Loading previous conversation state not yet implemented.')
        return None
    
    @computed_field
    @cached_property
    def conversation(self) -> ConversationWrapper:
        '''Lazily creates a merged conversation of the base conversation and the
        new conversation if a saved state has been loaded.'''
        return ConversationWrapper(self.base_conversation, self.loaded_conversation)
    
    @computed_field
    @cached_property
    def vector_store(self) -> vectorstores.VectorStore:
        '''Lazy load the vector store.'''
        return vectorstores.__getattr__(self.vector_store_name)
    
    @classmethod
    def from_yaml(cls, file_path: pathlib.Path) -> 'LanguageModelSettings':
        '''Load settings from a YAML file'''
        with open(file_path, 'r') as f:
            settings = yaml.safe_load(f)

        return cls(**settings)
