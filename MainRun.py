from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Load image
image_path = "2.jpg"
image = cv2.imread(image_path)

# Roboflow model prediction
rf = Roboflow(api_key="GBXneGBgt2OeLyBC5BSJ")
project = rf.workspace().project("logo-detector-cgxef")
model = project.version(2).model

result = model.predict(image_path, confidence=40, overlap=30).json()

# Extract detections from result
boxes = []
confidences = []
class_names = []

for pred in result["predictions"]:
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes.append([x1, y1, x2, y2])
    confidences.append(pred["confidence"])
    class_names.append(pred["class"])

# Map class names to numeric IDs
unique_classes = list(set(class_names))
class_name_to_id = {name: idx for idx, name in enumerate(unique_classes)}
class_ids = [class_name_to_id[name] for name in class_names]

# Convert to supervision.Detections object
detections = sv.Detections(
    xyxy=np.array(boxes),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

# Annotate
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

# Use original string labels for annotation
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=class_names)

# Display or save result
sv.plot_image(image=annotated_image, size=(16, 16))
