import json
import re
import string

from transformers import Conversation

class ConversationWrapper(Conversation):
    '''Wrapper around HuggingFace conversation class to be able to separate
    out new conversation from base conversation to allow appropriate squashing.
    Also allows for the loading of previous conversation states.
    
    Parameters:
        base_conversation (`Conversation`):
            The base conversation to start from.
        new_conversation (`Conversation`):
            Recorded conversation between user and assistant.
    '''
    def __init__(self, base_conversation: Conversation, new_conversation: Conversation = None):
        self._base_conversation = base_conversation
        self._conversation = new_conversation

        if self._conversation is None:
            self._conversation = Conversation()

        super().__init__(self._conversation)
        self.messages = self._base_conversation.messages + self._conversation.messages

    @property
    def base_conversation(self) -> Conversation:
        '''Return base conversation messages'''
        return self._base_conversation
    
    @property
    def conversation(self) -> Conversation:
        '''Return exchanged conversation between user and assistant'''
        return self[len(self._base_conversation):]
    
    @property
    def last_response(self) -> str:
        '''Return the last response from the assistant'''
        for message in reversed(self.messages):
            if message['role'] == 'assistant':
                return message['content']
            
        return ''
            
    @property
    def last_message(self) -> str:
        '''Return the last message from the user'''
        return '{role}: {content}'.format(**self.messages[-1])
            
    # TODO: Add summarisation based squashing
    def squash(self, home_state: dict):
        '''Squash the conversation'''
        if len(self.conversation) <= 4:
            return
        
        # TODO: Add a way to customise this
        home_state_str = json.dumps(home_state, indent='\t')
        update_state = Conversation([
            {
                'role': 'user',
                'content': 'Hi, please use this version of the model now:\n' \
                f'```json\n{home_state_str}\n```\nThat\'s the up to date version.'
            },
            {
                'role': 'assistant',
                'content': 'Ok sure, we can use that model state now.'
            }
        ])
        
        self._conversation = self.conversation[-4:]
        self.messages = self.base_conversation.messages + update_state.messages \
                        + self.conversation.messages
        
        return 'Conversation squashed successfully.'
        
    def reset(self):
        '''Resets conversation to base_conversation'''
        self.messages = self.base_conversation.messages
        self._conversation.messages = []
        return 'Chat reset successfully.'
        
    @staticmethod
    def check_for_command(prompt: str) -> str:
        '''Check for a command in the string'''
        # TODO: Maybe add more sophisticated command checking e.g. with BERT
        clean_prompt = re.sub(f'[{re.escape(string.punctuation)}]', '', prompt).lower().strip()
        if clean_prompt.startswith('squash chat please'):
            return 'squash'
        elif clean_prompt.startswith('reset chat please'):
            return 'reset'
        else:
            return None
