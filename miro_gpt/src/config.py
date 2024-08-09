def load_api_key():
    with open('/home/omar/pkgs/mdk-230105/catkin_ws/src/miro_willow/miro_gpt/src/api_key.txt') as f:
        key = f.readline()
        return key 
