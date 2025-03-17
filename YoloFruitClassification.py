from ultralytics import YOLO
import cv2
from PIL import Image


model = YOLO("best.pt")

image_path = "Orange.jpg"
image = Image.open(image_path)


results = model(image)

annotated_image = results[0].plot()

annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
model = YOLO("best.pt")

cv2.imshow("Fruit Classification", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
