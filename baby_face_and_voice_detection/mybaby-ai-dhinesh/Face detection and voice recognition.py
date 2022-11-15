
import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
  def __init__(self):
    self.known_face_encodings = []
    self.known_face_names = []

    # Resize frame for a faster speed
    self.frame_resizing = 0.25

  def load_encoding_images(self, images_path):
       
    # Load Images
    images_path = glob.glob(os.path.join(images_path, "*.*"))

    #print("{} encoding images found.".format(len(images_path)))

    # Store image encoding and names
    for img_path in images_path:
      img = cv2.imread(img_path)
      rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Get the filename only from the initial file path.
      basename = os.path.basename(img_path)
      (filename, ext) = os.path.splitext(basename)
      # Get encoding
      img_encoding = face_recognition.face_encodings(rgb_img)[0]

      # Store file name and file encoding
      self.known_face_encodings.append(img_encoding)
      self.known_face_names.append(filename)
    #print("Encoding images loaded")

  def detect_known_faces(self, frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Find all the faces and face encodings in the current frame of video
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
      # See if the face is a match for the known face(s)
      matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
      name = "Unknown"

    
      face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = self.known_face_names[best_match_index]
      face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
    face_locations = np.array(face_locations)
    face_locations = face_locations / self.frame_resizing
    return face_locations.astype(int),face_names

def facerecognition(database):#,video):
  import cv2
  import time
#from simple_facerec import SimpleFacerec
  videofacenames=[]
# Encode faces from a folder
  sfr = SimpleFacerec()
  sfr.load_encoding_images(database)

# Load Camera
  cap = cv2.VideoCapture(0)
  start_time = time.time()

  while True:
    ret, frame = cap.read()
    time_sec = time.time()
    if time_sec > start_time + 5:

      
        if len(videofacenames):
            print( max(set(videofacenames), key = videofacenames.count)) 
        else:
            print('No face found')
        #time_sec
        videofacenames = []
        start_time = time.time()
    #print(time)
  # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
      y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

      cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
      #print(face_names)
      videofacenames.append(name)
      
      key = cv2.waitKey(1)
      if key == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
      break
    # cv2.imshow(frame)
    #cv2.imwrite('/content/drive/MyDrive/work/finalwork/vidinf/im'+str(i)+'.jpg',frame)
    #i+=1
  # cv2.imshow('Webcam', frame)
  cv2.waitKey(1)

  #return videofacenames

#database="/home/mareena/Desktop/mybabyai/mybaby/image"
#video='/home/mareena/Desktop/mybabyai/mybaby/video/ajwa.mp4'

#facerecognition(database)#,video)

def extract_mfcc(filename):
    import librosa
    import numpy as np
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def audiolive():
    import pyaudio
    import wave
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "new_baby_voice.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, 
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def audio(filepath,modelfile):
    import joblib
    import librosa
    model=joblib.load(modelfile)
    extracted=extract_mfcc(filepath)
    Final_predictions=model.predict(extracted.reshape(1, -1))
    if Final_predictions==0:
        Final_predictions='Crying'
    elif Final_predictions==1:
        Final_predictions='Laughing'
    else:
        Final_predictions='Noise'
    return Final_predictions



from threading import Thread
import cv2
import numpy as np
import os
database="/home/mareena/Desktop/mybabyai/mybaby/image"
def facerec(): 
    facerecognition(database)
def voicerec():
    import librosa
    import time
    modelfile=r"/home/mareena/Desktop/mybabyai/mybaby/RF_uncompressed.joblib"
    while True:
        audiolive()
        prediction = audio("new_baby_voice.wav",modelfile)
        print("="*20)
        with open('readme.txt', 'w') as f:
            f.write('readme')
        #print(prediction)
        print("="*20)


Thread(target = facerec).start() 
Thread(target = voicerec).start()
