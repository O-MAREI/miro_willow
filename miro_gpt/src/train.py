import openai 
import json

class PredictResponse:
    """
    Functionality for Prompt Engineering to get a response of actions for Animal Mode.

    Parameters:
        self.api_key (str): GPT API Key
        self.mood (str): mood from previous dialog
    """
    def __init__(self):
        self.api_key = "sk-ioavVyrLfvfTJXXzeX_0bWjMDP4afEhrykK7kSwfTET3BlbkFJsHzJimL-PWV8YkXb-p9fSs_6yj7WM2Umsie_piixMA"
        self.mood = "no previous dialog"

    def predict(self, command):
        """
        Given a user voice statement converted to text, uses prompt engineering
        for ChatGPT to return a list of commands for the MiRo to perform. This function
        is what is called by the other functionality.
  
        Parameters:
            command (str): command input
        Returns:
            response_list (list): list of commands for MiRo to perform, following is an example:

            ["positive",[["lights", 0], ["move", 0.35, 0.21], ["tail_wag", 2, 2]],]
        """
        print("Generating GPT Response...")
        prompt = self._generate_prompt(command)



        
        valid_response = False
        while not valid_response:
            full_response = self._get_response(prompt)
     
            lines = full_response.split('\n', 1)
            print("LINES: ", lines)
            if len(lines) != 2:
                valid_response = False
                prompt = prompt + " Your last response did not declare the mood at all or in the right place, please fix."
                print("Retrying response... Mood was not outputted correctly")
                continue
               
            self.mood = lines[0]
            list_response = lines[1] 

            try:
                response_list = json.loads(list_response)
                if isinstance(response_list, list):
                    valid_response = True
            except json.JSONDecodeError:
                valid_response = False
                prompt = prompt + " Your last response did not return a Python List of the correct syntax. Please fix."
                print("Retrying response... Python list was not of correct format")
                continue


        response_list = json.loads(list_response)

        return response_list

    def _generate_prompt(self, command):
        """
        Generates the prompt to be sent to ChatGPT by using the mood from the previous dialog and the user command
  
        Parameters:
            command (str): command input
        Returns:
            gpt_prompt (str): input to be sent to ChatGPT
            
        """
        gpt_prompt = '''You are a Miro robot with language understanding and are going to receive a command from a human, which has been converted from voice to text. You are being used as a therapy robot in a health and social care setting, so perform an emotional response to give a sense that you are a real animal.
        The command is: {command} and your mood from the previous dialog is {mood}

        In order to deliver this emotional response, do the following:
        - Make a list based on the movement action items below.
        - Add these actions to the list in chronological order, of when they should be displayed in the emotional response.
        - If you want any actions to happen simultaneously, put these actions into their own list, inside this existing response list. This doesn't apply to pause and should be separate.
            The following example is the way to do the entries, dont copy this, there are so many more actions. This is just a template
            [[["move", 2, 0.2], ["head_lift", 50, 1],["head_yaw", 20, 1]], [["left_eye_close", true, 1]],[["move", -2, 0.2], ["head_lift", 20, 1]]]
        - Make sure that necessary actions happen simultaneously by putting them in the internal lists; otherwise, the robot looks too fake.
        - You can have up to 4 actions happening simultaneously. Every entry in the list should have a MINIMUM of 2 simultaneous actions; There must be a minimum of 4 entries for every response and a maximum of 6.
        - It should never look too still; ensure drastic adjustments to the head yaw, lift, and pitch, for example big angle changes in the head. Make sure the robot keeps looking up so it appears to be looking at the human talking to it.
        - You don't have to use all of the actions, only the ones that are relevant to the emotional response.
        - Make sure you base the response on the mood from the previous dialog. Add the new mood as the first line of the response NOT IN THE LIST BUT ABOVE IT, if there is no previous response add no mood instead.
        - You can repeat different actions to make the response longer and more animal-like but make sure there is lots of variety and consecutive actions aren't the same.
        - You can assume that the robot has working obstacle avoidance.
        - Make sure the response is SMOOTH, and each entry leads on well to the next. Remember, it needs to be as animal dog-like as possible.
        - Assume the person giving the command is directly in front of the MiRo. For example, if it's scared or sad it could move back but not necessarily; make sure all actions are tailored to the command received.

        
        Once again MAKE SURE THERE IS LOTS OF VARIETY OF ACTIONS and loads of movement, ensure its not the same movements every time.
         
        Check your answer 2 times, make sure it fully meets everything provided.
        
        Make sure you are returning the full Python list in the correct syntax, closing brackets, and using commas in the right place.
        [
            action lists.
        
        ]

        Ensure you have the new mood on first line OUTSIDE THE LIST

        Do not add any comments.
        
        ["move", Linear movement speed double between -0.35 for backwards and 0.35 for forwards, angular movement speed double between -0.35 and 0.35 for anti-clockwise and clockwise turning used for spinning around]
        ["head_pitch", degrees to move head -22 (head up) to 8 (head down), number of times to repeat this] Head pitch tilts the actual head up and down.
        ["head_lift", degrees to move head 8 (head up) to 60 (head down), number of times to repeat this] Head lift tilts the head and whole neck up and down. Much more noticeable than pitch.
        ["head_yaw", degrees to move head -90 (left) to 90 (right), number of times to repeat this] Head yaw rotates head.
        ["tail_wag", number of times to vertical wag, number of times to horizontal wag]  Horizontal for happy, vertical for sad.
        ["ears_left_rotate, 1 or 0] use ears when it's related to happy. Rarely use this
        ["ears_right_rotate", 1 or 0] use ears when it's related to happy. Rarely use this
        ["lights", 0, 1 or 2] It sould be 0 for positive emotions. It should be 1 for negative emotions. It should be 2 for neutral emotions.

        "lights" MUST BE INCLUDED. It MUST be the first action in the first simultaneous list. PLEASE don't forget that.

        IMPORTANT: Here is how each action's values should be based on emotion:
        Use the following as a guideline, you don't have to follow it exactly:
        "move" - You should move forward for positive emotions. You should move backwards for negative emotions.
        "tail_wag" - Wag the tail horizontally for positive emotions. DO NOT wag tail for negative emotions.
        "ears_left_rotate" - You should rotate for positive emotions. You should not rotate for negative emotions.
        "ears_right_rotate" - You should rotate for positive emotions. You should not rotate for negative emotions.
        "head_lift" - You are more likely to lift the head up when the emotion is positive, and more likely to lower the head down when the emotion is negative
        '''

        return gpt_prompt.format(command=command, mood=self.mood)

    def _get_response(self, prompt):
        """
        Gets the response from ChatGPT using gpt-3.5-turbo
  
        Parameters:
            prompt (str): the formatted prompt
        Returns:
            chat (str): the return string of actions
            
        """
        openai.api_key = self.api_key
        messages = [{"role": "user", "content": prompt}]
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=messages
        )

        return chat['choices'][0]['message']['content'].strip()

def main():
    predict = PredictResponse()
    print(predict.predict("Miro i love you, spin around"))

if __name__ == "__main__":
    main()
