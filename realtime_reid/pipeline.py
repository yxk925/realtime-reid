import os

import cv2
import numpy as np
import logging
try:
    import rospy
except Exception:
    rospy = None

# Module-level logger: set to DEBUG to emit detailed pipeline debug information.
# Keep handlers configuration to the application; here we only set the level.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()               # outputs to stderr
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
from .classifier import PersonReID
from .feature_extraction import PersonDescriptor
from .person_detector import PersonDetector
from .visualization_utils import color


class Pipeline:
    def __init__(self,
                detector: PersonDetector = PersonDetector(),
                descriptor: PersonDescriptor = PersonDescriptor(),
                classifier: PersonReID = PersonReID()) -> None:
        """Initialize the pipeline by creating the necessary objects."""

        self.detector = detector
        self.descriptor = descriptor
        self.classifier = classifier

    def process(
        self,
        msg: bytes | np.ndarray,
        save_dir: str = None,
        return_bytes: bool = False
    ) -> np.ndarray | bytes:
        """
        Process the input message by detecting and identifying persons
        in the image.

        Parameters
        ----------
            msg (Message), bytes | np.ndarray, required
                The input message containing the image data.
            save_dir, str, default None
                The directory to save the detected images.
                Leave it empty if you don't want to save the images.
            return_bytes, bool, default False
                Whether to return the processed image as bytes.

        Returns
        -------
            Image: The processed image with bounding boxes and labels for
        detected persons.
        """
        if not isinstance(msg, bytes) and not isinstance(msg, np.ndarray):
            raise TypeError(f"msg must be of type bytes or numpy array. Got {type(msg)}.")

        detected_data = self.detector.detect(msg)
        logger = logging.getLogger(__name__)
        # Report logger availability and level via ROS debug if rospy is present
        try:
            level_name = logging.getLevelName(logger.getEffectiveLevel())
        except Exception:
            level_name = 'UNKNOWN'
        if rospy is not None:
            try:
                rospy.logdebug("Logger '%s' available. Level: %s", logger.name, level_name)
            except Exception:
                # Fallback to the standard logger if rospy logging fails
                logger.debug("rospy.logdebug failed; Logger '%s' level: %s", logger.name, level_name)
        else:
            logger.debug("rospy not available. Logger '%s' level: %s", logger.name, level_name)
        logger.debug("detected_data type: %s", type(detected_data))
        logger.debug("detected_data value: %s", detected_data)
        if isinstance(detected_data, list):
            logger.debug("list length: %d", len(detected_data))
            if len(detected_data) > 0:
                logger.debug("first element type: %s", type(detected_data[0]))
                logger.debug("first element shape: %s", getattr(detected_data[0], 'shape', 'N/A'))
        
        
        # Convert the image data to an array
        #image_data = np.frombuffer(msg, dtype=np.uint8)
        #final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        final_img = msg.copy()
        logger.debug("final_img type: %s", type(final_img))
        
        ids = []
        if len(detected_data) >= 1:
            detected_data = detected_data[0]
        else:
            return final_img, [], ids
        
        for detected_box in detected_data.boxes:
            # detected_box.xyxy is a (1, 4) tensor
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            cropped_img = final_img[ymin:ymax, xmin:xmax, :]

            # Debug: log detected_box and cropped image info
            try:
                logger.debug("detected_box.xyxy: %s", xyxy)
            except Exception:
                logger.debug("detected_box: %s", detected_box)
            try:
                logger.debug("cropped_img shape: %s dtype: %s", cropped_img.shape, cropped_img.dtype)
            except Exception:
                logger.debug("cropped_img info unavailable")

            # A small solution for #3 (initial partial visibility)
            # This is a temporary solution, but it works beautifully.
            # Solution: Check if the person is fully visible
            offset = 2  # because the bbox is not (always) accurate
            lower_bound = ((xmin - offset) <= 0 or (ymin - offset) <= 0)
            upper_bound = (xmax + offset >= final_img.shape[1]
                           or (ymax + offset) >= final_img.shape[0])
            if lower_bound or upper_bound:
                current_id = -1
                logger.debug("person partially visible (bounds): lower=%s upper=%s => assign id -1", lower_bound, upper_bound)
                current_person = None
            else:
                current_person = self.descriptor.extract_feature(cropped_img)
                # Debug: log basic info about extracted feature
                try:
                    if hasattr(current_person, 'shape'):
                        logger.debug("current_person feature shape: %s dtype: %s", current_person.shape, current_person.dtype)
                    else:
                        logger.debug("current_person feature type: %s", type(current_person))
                except Exception:
                    logger.debug("current_person info unavailable")

                current_id = self.classifier.identify(
                    target=current_person,
                    do_update=True
                )
            ids.append(current_id)

            # Save the cropped image before drawing the bounding box
            if save_dir is not None:
                save_filename = f"{len(os.listdir(save_dir))}_{current_id}"
                cv2.imwrite(
                    f"{save_dir}/{save_filename}.jpg",
                    cropped_img
                )

        for detected_box, current_id in zip(detected_data.boxes, ids):
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            # Draw bounding box and label
            label = f"{current_id}"
            # Debug: log drawing parameters
            try:
                draw_color = color.create_unique_color(current_id)
            except Exception:
                draw_color = None
            logger.debug("Drawing box id=%s bbox=%s color=%s", current_id, (xmin, ymin, xmax, ymax), draw_color)
            cv2.rectangle(
                img=final_img,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=draw_color if draw_color is not None else color.create_unique_color(current_id),
                thickness=2,
            )
            cv2.putText(
                img=final_img,
                text=label,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=draw_color if draw_color is not None else color.create_unique_color(current_id),
                thickness=2,
            )
            logger.debug("Drew label '%s' for id=%s at pos=%s", label, current_id, (xmin, ymin))

        if return_bytes:
            return cv2.imencode(
                '.jpg',
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )[1].tobytes()

        return final_img, detected_data.boxes, ids
