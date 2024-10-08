import os
import time

import cv2
from kafka import KafkaProducer


class VideoProducer:
    def __init__(
            self,
            topic: str,
            interval: float,
            bootstrap_servers: str = 'localhost:9092'):

        self.INTERVAL = interval
        # Incase user input FPS instead of interval
        if self.INTERVAL > 1:
            self.INTERVAL = 1 / self.INTERVAL

        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.TOPIC = topic

    def encode_and_produce(self, frame):
        frame = self.process_frame(frame)

        # Convert image to jpg format
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to Kafka
        self.producer.send(self.TOPIC, buffer.tobytes())

    def publish_from_video(self, source):
        video = cv2.VideoCapture(source)            

        # Set default interval for video to video FPS
        if self.INTERVAL == -1:
            self.INTERVAL = 1 / video.get(cv2.CAP_PROP_FPS)

        last_time = -1
        while video.isOpened():
            success, frame = video.read()

            # Ensure file was read successfully
            if not success:
                print("bad read!")
                break
            
            if last_time < 0 or (time.time() - last_time) > self.INTERVAL:
                self.encode_and_produce(frame)
                last_time = time.time()
            
            cv2.waitKey(1)
            
        video.release()
        return True

    def publish_from_img_folder(self, source_path):
        # Open folder
        image_files = [f for f in os.listdir(source_path)
                       if f.endswith(('.jpg', '.png'))]

        # Set default interval for image folder to 12 FPS
        if self.INTERVAL == -1:
            self.INTERVAL = 1 / 12

        for img in image_files:
            image_path = os.path.join(source_path, img)
            frame = cv2.imread(image_path)

            self.encode_and_produce(frame)
            time.sleep(self.INTERVAL)
        return True

    def publish_video(self, source: str):
        """
        Publish given video file to `self.topic`.
        There are 2 possible `video_path` input, a link to a video
        (a file) or a path to a folder that contains of images.

        Parameters
        ----------
        `source`: str
            path to video file (camera demo)
        """
        try:
            if os.path.isfile(source):
                print(f"Publish from video {source} to topic {self.TOPIC}")
                self.publish_from_video(source)
            elif os.path.isdir(source):
                print(f"Publish from folder {source} to topic {self.TOPIC}")
                self.publish_from_img_folder(source)
            else:
                print(f"Publish from source {source} to topic {self.TOPIC}")
                source_int = int(source)
                self.publish_from_video(source_int)
            print('Publish complete!')
        except KeyboardInterrupt:
            print("Publish stopped.")

    @staticmethod
    def process_frame(frame):
        # Image with fixed size, reserve aspect ratio
        original_ratio = frame.shape[1] / frame.shape[0]
        width = 640
        height = int(width / original_ratio)
        frame = cv2.resize(frame, (width, height))

        return frame
