#!/usr/bin/env python3
import numpy as np
import os
import rospy # ROS Python interface
from std_msgs.msg import Int16MultiArray, Bool, String # ROS message for mics
import time
import pvporcupine
import openai, config
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
from std_msgs.msg import Int16MultiArray, UInt32MultiArray, Float32MultiArray, UInt16MultiArray
from sensor_msgs.msg import JointState
from elevenlabs.client import ElevenLabs
from elevenlabs import save, Voice, VoiceSettings
import pvcobra
from miro_movement5 import Movement

# testing purposes
import wave, struct
import threading

SAMPLE_COUNT = 640
SAMPLING_TIME = 10 # in terms of seconds
class NamedBytesIO(BytesIO):
        def __init__(self, name, data=None):
            super().__init__(data if data is not None else b'')
            self.name = name


class ProcessAudio(object):

    # update the detected words with the messages being published
    def __init__(self):
        
        # init for the processing audio
        rospy.init_node("process_audio")
        openai.api_key = config.load_api_key()
        self.mic_data = np.zeros((0, 4), 'uint16')      # the raw sound data obtained from the MiRo's message.
        self.micbuf = np.zeros((0, 4), 'uint16')        # the raw sound data that is has been recorded for use. This has been recorded and passes to be process when it has enough sample count.
        self.detected_sound = np.zeros((0,1), 'uint16') # the detected sound throughout.
        self.recorded_sound = np.zeros((0,1), 'uint16') # to store the first few seconds of sound to be init when hearing a response.
        self.to_record = np.zeros((0,1), 'uint16')      # the frame which is recorded for use in chatgpt.
        self.zcr_frame = np.zeros((0,1), 'uint16')      # zero crossing rate frame.
        self.process_whisper_msg = Bool()               # message used to let whisper know when to start processing.
        self.gesture_msg = String()                     # gestures message
        self.stop_record = time.time() - 5              # the time when the robot should stop recording. The robot will stop recording 3 seconds after it hears "Hey MiRo" and the user stops speaking after.
        self.start_check_time = time.time() + 1.5       # the time when the robot itself is speaking.
        self.color = UInt32MultiArray()
        self.color.data = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]
        self.hey_miro = False
        self.switch_modes = False
        self.silence_count = 0
        self.in_conversation = False            
        self.detected = True
        self.waiting_count = 0
        self.talking = False
        self.there_is_voice = False

        self.cos_cmd = Float32MultiArray()
        self.cos_cmd.data = [0,0.5  ,0,0,0.3333,0.3333] # default for the eye (0.3333)
        self.joint_cmd = JointState()
        self.joint_cmd.position = [0,0,0,0]
        self.color_change = UInt32MultiArray()


        # Tone Control
        self.tone = UInt16MultiArray()
        self.tone.data = [0, 0, 0]
        self.beep = False

        # Animal/LLM
        self.state = "llm"

        # Whisper
        self.message = String()

        # ChatGPT
        self.model = "gpt-3.5-turbo"
        self.message_history =[]
        system_message = {"role": "system", "content": "You are a friendly robot assistant called MiRo."}
        self.message_history.append(system_message)

        # Miro Response
        self.data = []

        # AI voice
        self.client = ElevenLabs(
            api_key = "63e1b00f0bcd8e1af8fecffa7693781e"#"fe78b860a763919b03861b0ef7f4ec0f"
        )


        # porcupine access
        self.access_key = "+EZod1b3D0XXpfxZLOnsyowiaMGYwRwSlO6bnPoMKRo2EJlXQkPPOA=="
        # new_path = "../pkgs/mdk-210921/catkin_ws/src/speech_recognition_porcupine/src"
        # os.chdir(new_path)
        self.handle = pvporcupine.create(access_key=self.access_key, 
									keywords=['hey google'],
									keyword_paths=['/home/student/pkgs/mdk-230105/catkin_ws/src/miro_gpt/src/processed_data/Hey-Miro_en_linux_v3_0_0.ppn',
                                                   '/home/student/pkgs/mdk-230105/catkin_ws/src/miro_gpt/src/processed_data/Switch-Modes_en_linux_v3_0_0.ppn'])
        
        #self.switch = pvporcupine.create(access_key=self.access_key, 
		#							keywords=['hey google'],
		#							keyword_paths=['/home/student/pkgs/mdk-230105/catkin_ws/src/miro_gpt/src/processed_data/Switch-Modes_en_linux_v3_0_0.ppn'])
        
        self.cobra = pvcobra.create(access_key=self.access_key)
        # self.cobra._frame_length = 1000

        # ros subcribers and publishers to be used
        self.topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        self.subscriber = rospy.Subscriber(
            self.topic_base_name + "/sensors/mics", Int16MultiArray, self.audio_cb, tcp_nodelay=True
        )
        # self.whisper_publisher = rospy.Publisher(
        #     self.topic_base_name + "/gpt_speech/process_whisper", Bool, queue_size=0
        # )
        self.subscriber_check_gtts = rospy.Subscriber(
            self.topic_base_name + "/control/stream", Int16MultiArray, self.check_gtts_cb, tcp_nodelay=True
        )
        self.pub_gestures = rospy.Publisher(
            self.topic_base_name + "/gpt_speech/actions", String, queue_size=0
        )
        # Response publisher
        self.pub_response = rospy.Publisher(self.topic_base_name + "/gpt_speech/prompt_response", Int16MultiArray, queue_size=0)

        self.pub_stream = rospy.Publisher(self.topic_base_name + "/control/stream", Int16MultiArray, queue_size=0)


        # movement for either tilt, lift, yaw or pitch
        self.pub_kinematic = rospy.Publisher(
            self.topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )

        # movement for the tail, eye lid, ears
        self.pub_cosmetic = rospy.Publisher(
            self.topic_base_name + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )

        # color of lights on MiRo (r,g,b)
        self.pub_illumination = rospy.Publisher(
            self.topic_base_name + "/control/illum", UInt32MultiArray, queue_size=0
        )

        # Tone publisher
        self.pub_tone = rospy.Publisher(self.topic_base_name + "/control/tone", UInt16MultiArray, queue_size=0)
                                
    def is_speech(self, pcm_data, sample_rate=16000):
        """Check if the provided PCM data frame contains speech. Assumes 16-bit mono."""
        return self.vad.is_speech(pcm_data.tobytes(), sample_rate)

    # callback function that updates the time when it sees a message be published
    def check_gtts_cb(self, msg):
        self.gesture_msg.data = "normal"
        self.pub_gestures.publish(self.gesture_msg)
        self.start_check_time = time.time()

    def set_move_cosmetic(self, tail_pitch = 0, tail_yaw = 0, left_eye = 0, right_eye = 0, left_ear = 0.3333, right_ear = 0.3333):
        self.cos_cmd.data = [tail_pitch,tail_yaw,left_eye,right_eye,left_ear,right_ear]

    def set_move_kinematic(self, tilt = 0, lift = 0, yaw = 0, pitch = 0):
        self.joint_cmd.position = [tilt, lift, yaw, pitch]
    
        # set color
    def get_illumination(self, red = 0, green = 0, blue = 0):
        # changing the rgb format into android format to be published in MiRo message
        color_detail = (int(red), int(green), int(blue))
        color = '0xFF%02x%02x%02x'%color_detail
        color = int(color, 16)
        return color
    
    def done(self):
        color = self.get_illumination(green = 150)
        self.color_change.data = [
            color,
            color,
            color,
            color,
            color,
            color
        ]

    def no_process(self):
        color = self.get_illumination()
        self.color_change.data = [
            color,
            color,
            color,
            color,
            color,
            color
        ]
    
    def loading_colors(self, *args):
        color = self.get_illumination(green=150)
        left_light = [color, color, 0, 0, 0, 0]
        centre_light = [0, 0, color, color, 0, 0]
        right_light = [0, 0, 0, 0, color, color]
        lights = [left_light, centre_light, right_light]
        while self.beep:
            for light in lights:
                self.color_change.data = light
                self.pub_illumination.publish(self.color_change)
                time.sleep(0.5)

    
    def make_tone(self, val_freq, val_volume, val_duration, *args):
        print("HERE: ", val_duration)
        freq, volume, duration = range(3)
        while self.beep:
            self.tone.data[freq] = val_freq
            self.tone.data[volume] = val_volume 
            self.tone.data[duration] = val_duration
            self.pub_tone.publish(self.tone)
            time.sleep(2)
        


    # process audio for wake word and the recording to be sent for speech to text
    def audio_cb(self, msg):
        # start recording only if the miro is not speaking
        if time.time() - self.start_check_time > 1.5:
            # reshape into 4 x 500 array
            data = np.asarray(msg.data, dtype=np.int16)
            # print(data)
            # vad = self.energy_vad(data)
            self.mic_data = np.transpose(data.reshape((4, 500)))
            self.record = False      
      
            if not self.micbuf is None:
                self.micbuf = np.concatenate((self.micbuf, self.mic_data))
                
                if self.micbuf.shape[0] >= SAMPLE_COUNT:
                    outbuf = self.micbuf[:SAMPLE_COUNT]
                    self.micbuf = self.micbuf[SAMPLE_COUNT:]
                    
                    # get audio from left ear of Miro
                    detect_sound = np.reshape(outbuf[:, [1]], (-1))
                    for i in range(1,3):
                        detect_sound = np.add(detect_sound, np.reshape(outbuf[:,[i]], (-1)))

                    # downsample to sampling rate accepted by picovoice
                    outbuf_dwnsample = np.zeros((int(SAMPLE_COUNT / 1.25), 0), 'uint16')
                    i = np.arange(0, SAMPLE_COUNT, 1.25)
                    j = np.arange(0, SAMPLE_COUNT)
                    x = np.interp(i, j, detect_sound[:])
                    outbuf_dwnsample = np.concatenate((outbuf_dwnsample, x[:, np.newaxis]), axis=1)
                    outbuf_dwnsample = outbuf_dwnsample.astype(int)
                    outbuf_dwnsample = np.reshape(outbuf_dwnsample[:, [0]], (-1))
                    if len(self.recorded_sound) < 20000:
                        self.recorded_sound = np.append(self.recorded_sound, detect_sound)
                    else:
                        self.recorded_sound = np.append(self.recorded_sound[20000:], detect_sound)
                    
                    # check for any wake words
                    keyword_index = self.handle.process(outbuf_dwnsample)
                    #switch_modes_index = self.switch.process(outbuf_dwnsample)

                    voice_probability = self.cobra.process(outbuf_dwnsample)

                    if self.hey_miro:
                        if voice_probability > 0.6:
                            self.silence_count = 0
                        else:
                            self.silence_count += 1


                    self.print = keyword_index
                    # if the wake word is "Hey MiRo" start recording
                    if keyword_index != -1 and not self.in_conversation and not self.hey_miro:
                        if keyword_index == 0:

                            print("Detected: Hey Miro!")
                            self.hey_miro = True
                            # self.gesture_msg.data = "notice"
                            self.set_move_kinematic(lift=0.399, pitch=-0.21)
                            self.done()
                            
                            self.pub_kinematic.publish(self.joint_cmd)
                            self.pub_illumination.publish(self.color_change)
                        
                            self.detected_sound = np.zeros((0,1), 'uint16')

                            self.stop_record = time.time()
                            self.silence_count = 0

                        else:

                            print("Detected: Switch Mode!")
                            self.state = "animal" if self.state == "llm" else "llm"
                            self.set_move_kinematic(lift=1.047192, pitch=0.1396256)
                            self.pub_kinematic.publish(self.joint_cmd)
                            time.sleep(3)
                            self.set_move_kinematic(lift=0.59)
                            self.pub_kinematic.publish(self.joint_cmd)
                            print("THIS IS THE MODE LOL: ", self.state)

                    #if switch_modes_index != -1 and not self.in_conversation and not self.hey_miro:
                    #    print("Detected: Switch Mode!")
                    #    self.state = "animal" if self.state == "llm" else "llm"
                    #    self.set_move_kinematic(lift=1.047192, pitch=0.1396256)
                    #    self.pub_kinematic.publish(self.joint_cmd)
                    #    time.sleep(3)
                    #    self.set_move_kinematic(lift=0.59)
                    #    self.pub_kinematic.publish(self.joint_cmd)
                    #    print("THIS IS THE MODE LOL: ", self.state)
                    #    self.detected_sound = np.zeros((0,1), 'uint16')

                    #    self.stop_record = time.time()
                    #    self.silence_count = 0

                    

                    if self.hey_miro and self.detected:
                        self.record = True
                        self.detected_sound = np.append(self.detected_sound, self.recorded_sound)
                        self.recorded_sound = np.zeros((0,1), 'uint16')
                        if not self.in_conversation:
                            print("HERE: ", self.silence_count)
                            if self.silence_count > 60:
                                self.to_record = self.detected_sound
                                print("The length of recording: ",len(self.to_record))
                                self.detected_sound = np.zeros((0,1), 'uint16')
                                self.silence_count = 0
                                self.detected = False
                                self.in_conversation = True
                                
                        else:
                            if not self.talking:
                                self.waiting_count += 1
                            if self.waiting_count >= 60:
                                self.detected_sound = np.append(self.detected_sound, self.recorded_sound)
                                self.recorded_sound = np.zeros((0,1), 'uint16')
                                print("DONE")
                                self.cobra._frame_length = len(self.detected_sound)
                                probability = self.cobra.process(self.detected_sound)
                                print(probability)
                                if  probability > 0.6:
                                    self.set_move_kinematic(lift=0.59,yaw=-0.8)
                                    self.done()
                                    self.pub_kinematic.publish(self.joint_cmd)
                                    self.pub_illumination.publish(self.color_change)
                                    
                                    print("VOICED")

                                    self.silence_count = 0
                                    self.waiting_count = 0
                                    self.there_is_voice = True
                                else:
                                    if self.there_is_voice:
                                        self.beep = True
                                        print("Sending recording..")
                                        self.to_record = self.detected_sound
                                        print("The length of recording: ",len(self.to_record))
                                        self.detected_sound = np.zeros((0,1), 'uint16')
                                        self.silence_count = 0
                                        self.detected = False
                                        self.waiting_count = 0
                                        self.there_is_voice = False

                                    else:
                                        print("NOT VOICED")
                                        # reset MiRo
                                        self.set_move_kinematic(lift=0.59)
                                        self.no_process() # Resetting the colours
                                        self.tone.data[0] = 2 # Frequency
                                        self.tone.data[1] = 75 # Volume
                                        self.tone.data[2] = 1 # Duration
                                        self.pub_tone.publish(self.tone)
                                        self.pub_kinematic.publish(self.joint_cmd)
                                        self.pub_illumination.publish(self.color_change)

                                        self.waiting_count = 0
                                        self.in_conversation = False
                                        self.hey_miro = False
                                        self.there_is_voice = False
                                        self.message_history = [{"role": "system", "content": "You are a friendly robot assistant called MiRo."}]
                                        self.beep = False

                            self.cobra._frame_length = 512   

    
    
    # to be looped. the method is used for saving audio files and publishing message to whisper that it is ready to be processed.
    # To-Do: write to Bytes.ioprocess_audio.py
    def record_audio(self):
        if len(self.to_record) > 0:
            self.beep = True
            self.sound = threading.Thread(target=self.make_tone, args=(1, 75, 1,))
            self.lights = threading.Thread(target=self.loading_colors, args=(self.beep, ))
            self.sound.start()
            self.lights.start()
            wav_io = NamedBytesIO("audio.wav")
            with wave.open(wav_io, 'wb') as file:
                file.setframerate(20000)
                file.setsampwidth(2)
                file.setnchannels(1)
                for s in self.to_record:
                    try:
                        file.writeframes(struct.pack('<h', s))
                    except struct.error as err:
                        pass
                        # print(err)
            wav_io.seek(0) # Rewind the buffer
            if self.hey_miro:
                self.whisper_process(wav_io)
                self.chatGPT_process()
                self.set_move_kinematic(lift=0.59)
                self.no_process()
                self.pub_kinematic.publish(self.joint_cmd)
                self.pub_illumination.publish(self.color_change)
                self.detected = True
                

            self.to_record = np.zeros((0,1), 'uint16')
        

    def whisper_process(self, audio_io):
        print("In Whisper")
        transcript = openai.Audio.transcribe("whisper-1", audio_io)
        self.message.data = transcript.text
        print("Result: ", self.message.data)
        self.beep = False
            
    def chatGPT_process(self):
        print("Processing prompt: ", self.message.data)
        user_prompt = {"role": "user", "content": self.message.data}
        self.message_history.append(user_prompt)

        if self.state == "llm":
            chat = openai.ChatCompletion.create(
                model=self.model,
                messages=self.message_history
            )
            print(self.message_history)
            reply = chat['choices'][0]['message']['content']
            print(f"Response from GPT: {reply}")

            # Resetting miro position
            self.set_move_kinematic(lift=0.59)
            self.no_process()
            self.pub_kinematic.publish(self.joint_cmd)
            self.pub_illumination.publish(self.color_change)

            self.text_to_wav(reply)
                
            rate = rospy.Rate(10)
            self.d = 0
            while not rospy.core.is_shutdown():
                if (self.d < len(self.data)):                
                    msg = Int16MultiArray(data = self.data[self.d:self.d + 1000])
                    self.d += 1000
                    self.pub_stream.publish(msg)
                else:
                    self.talking = False
                    break
                rate.sleep()
        elif self.state == "animal":
            movement = Movement(self.message.data)
            movement.main()
        
    
    def text_to_wav(self, reply):
        # mp3_fp = BytesIO()
        audio = self.client.generate(
            text = reply,
            voice = Voice(
                voice_id = "vGQNBgLaiM3EdZtxIiuY", #"jBpfuIE2acCO8z3wKNLl",
                settings = VoiceSettings(stability=0.5, similarity_boost=0.75)

            )
        )
        # Consume the generator to get the audio data
        audio_byte = b''.join(audio)

        # use BytesIO for in-memory processing
        audio_data = BytesIO(audio_byte)

        

        # res = gTTS(text = reply, lang = 'en', slow = False)
        # # res.save("response.mp3")
        # res.write_to_fp(mp3_fp)
        # mp3_fp.seek(0) # move the cursor to the beginning of the BytesIO buffer

        seg = AudioSegment.from_file(audio_data, format='mp3')
        # seg=AudioSegment.from_mp3("response.mp3")
        seg = seg.set_frame_rate(8000)
        seg = seg.set_channels(1)

        wav_io = BytesIO()
        seg.export(wav_io, format='wav')
        wav_io.seek(0) # Rewind the buffer for reading

        wav_io.seek(44) # Skip the WAV header (44 bytes)
        dat = np.frombuffer(wav_io.read(), dtype=np.int16) # read as int16
        wav_io.close()

        # normalise wav
        dat = dat.astype(float)
        sc = 32767.0 / np.max(np.abs(dat))
        dat *= sc
        dat = dat.astype(np.int16).tolist()
        
        self.data = dat
        self.d = 0
        self.talking = True
        

if __name__ == "__main__":
    main = ProcessAudio()
    while not rospy.core.is_shutdown():
        main.record_audio()
