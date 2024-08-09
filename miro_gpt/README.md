
# MiRoGPT


## Project Description

With the recent surge in Large Language Models (LLMs), it brings about the question of how they can be used in robotics, specifically in the MiRo.  However, limited research exists on how prompt engineering and ChatGPT can be used to simulate emotional responses in a Biomimetic Robot like MiRo. Hence, in this study, we build upon this previous work developed by Aung to evaluate two different ChatGPT implementations. The solution has two different modes which can be switched in between. The first, titled 'LLM Mode',  involves providing the MiRo with a human voice and decreasing the processing times of the current MiRo ChatGPT interface. The second, titled 'Animal Mode' explores using Prompt Engineering to simulate an emotional response in MiRo in response to voice commands. 

## Running the Project

The project uses the existing ROS middleware to control and communicate with the MiRo robot. This guide assumes that a MiRo has been connected to a laptop with a Linux Operating System.


The dependencies for the project should first be installed by running:
```bash
  pip install -r requirements.txt
```

Please note, the solution uses an earlier version of OpenAI, specifically 0.28.

To run the solution
```bash
  roslaunch miro_gpt miro_chatgpt.launch
```

To switch between both animal and LLM mode, use the audio command "Switch Modes".

The wake command for both modes is "Hey MiRo"

## Demo
To view a video demostration of the solution, please vist: https://www.youtube.com/watch?v=MSKHuTexpvk

## Authors

- [@louiswillison](https://github.com/louiswillison)
- [@jjtKenji](https://github.com/jjtKenji)
- [@aca20oya](@https://github.com/aca20oya)
- [@max-wilko1](https://github.com/Max-wilko1)

