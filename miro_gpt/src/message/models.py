from pydantic import BaseModel
from typing import Optional

class WISMessage(BaseModel):
    '''Validation class for messages from willow inference server (WIS)'''
    text: str
    language: Optional[str] = 'en'
