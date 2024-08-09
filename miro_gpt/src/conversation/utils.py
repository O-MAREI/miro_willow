import json

from transformers import Conversation


smart_home_data_example = {
    "bedroom_light": "off",
    "kitchen_light": "off",
    "living_room_light": "off",
    "heating_target_temperature": "20",
    "front_door": "locked",
}

def load_conversation(json_file: str, **str_kwargs) -> Conversation:
    '''Load the base conversation from a json file with JSON format:
    [{ 'role': '<role>', 'content': '<content>' }, ...]
    
    Parameters:
        json_file (`str`):
            The path to the json file containing the base conversation.
        str_kwargs (`dict`):
            Keywords to format into the conversation strings.
    '''
    with open(json_file, 'r') as f:
        conversation = json.load(f)

    if not isinstance(conversation, list):
        raise ValueError(f'The conversation in {json_file} must be a list of dictionaries.')
    
    # Add data example to kwargs for formatting
    str_kwargs |= {'smart_home_data_example': json.dumps(smart_home_data_example, indent='\t')}

    # Format content strings with str_kwargs
    for turn in conversation:
        if not turn.keys() == {'role', 'content'}:
            raise KeyError(f'Each conversation turn in {json_file} must have "role" and "content" as keys.')
        
        try:
            turn['content'] = turn['content'].format(**str_kwargs)
        except KeyError as e:
            raise KeyError(f'The content string in {json_file} could not be formatted with {str_kwargs}. '
                            'You may have forgotten to escape in-string JSON using {{}}') from e
        
    return Conversation(conversation)
