import argparse
from queue import Queue
import numpy as np
import cv2
import threading
import time
from kafka import KafkaConsumer
from realtime_reid.pipeline import Pipeline


img_count = 0

def parse_args():
    """Parse User's input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reid",
                        type=str,
                        choices=["y", "n", "spark"],
                        default="n",
                        help="Set this 'y' if you want to apply reid on the"
                        "images, 'spark' if you want to run the spark and")

    return parser.parse_args()


args = vars(parse_args())

reid_pipeline = Pipeline()


# Create a Queue to hold the processed images
processed_images = Queue()

def process_frame(frame):
    # Image with fixed size, reserve aspect ratio
    original_ratio = frame.shape[1] / frame.shape[0]
    width = 640
    height = int(width / original_ratio)
    frame = cv2.resize(frame, (width, height))

    return frame
    
def process_vedio(frame):
    print("process_vedio")
    frame = process_frame(frame)

    # Convert image to jpg format
    _, buffer = cv2.imencode('.jpg', frame)
    
    final_img = reid_pipeline.process(buffer)

    # Add the processed image to the Queue
    processed_images.put(("Standalone", final_img))

def vedio_thread_proc():
    
        video = cv2.VideoCapture(0)            

        INTERVAL = 0.033
        # Set default interval for video to video FPS
        if INTERVAL == -1:
            INTERVAL = 1 / video.get(cv2.CAP_PROP_FPS)

        last_time = -1
        while video.isOpened():
            success, frame = video.read()

            # Ensure file was read successfully
            if not success:
                print("bad read!")
                break
            
            now = time.time()
            #print("last_time:{0}, now:{1}".format(last_time, now))
            if last_time < 0 or (now - last_time) > INTERVAL:
                process_vedio(frame)
                last_time = time.time()
            
        video.release()

def start_threads():
    """Start processing messages from both topics using threads"""
    thread_0 = threading.Thread(
        target=vedio_thread_proc,
        args=()
    )

    thread_0.start()
    # thread_1.start()

    return thread_0


def display_images():
    """Display the processed images in the main thread"""
    while True:
        # Get the next processed image and display it
        consumer_name, final_img = processed_images.get()
        cv2.imshow(consumer_name, final_img)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def main():

    thread_0 = start_threads()
    display_images()

    # Wait for both threads to finish
    thread_0.join()
    # thread_1.join()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
