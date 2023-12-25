import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import time
import tkinter as tk
from tkinter import ttk


model = load_model('mobile_BILSTM_model.h5')
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
# SKIP_FRAMES = 5
classes = ['NonViolence', 'Violence']

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Violence Detection App")
        self.root.geometry("420x220")

        self.start_button = ttk.Button(self.root, text="Start Processing", command=self.predict_live_video, style='TButton' )
        self.start_button.pack(pady=10, padx=10)
        self.style = ttk.Style()
        self.style.configure('TButton', font=('calibri', 20, 'bold'), foreground='blue')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.configure(bg='black')
   
    def predict_live_video(self):
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        frames_list = []

        while True:
            frame = vs.read()
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

            if len(frames_list) == SEQUENCE_LENGTH:
                predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = classes[predicted_label]
                confidence = predicted_labels_probabilities[predicted_label]

                if predicted_class_name == "Violence":
                    cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
                else:
                    cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)
                
                cv2.putText(frame, str(round(confidence, 4)), (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 12)

                cv2.imshow("Live Feed", frame)

                frames_list.pop(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vs.stop()
        cv2.destroyAllWindows()
        
    def on_close(self):
        self.root.destroy()



if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
