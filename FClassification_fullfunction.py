import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

model = YOLO("best01.pt") 

def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    results = model(img)

    detections = results[0].names 
    detected_objects = results[0].boxes.data

    if detected_objects.shape[0] > 0:
        detected_class = int(detected_objects[0, 5])
        fruit_name = detections[detected_class]
        result_label.config(text=f"Detected Fruit: {fruit_name}", fg="green")
    else:
        result_label.config(text="No fruit detected!", fg="red")

    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

root = tk.Tk()
root.title("Fruit Classification")
root.geometry("500x500")

btn = tk.Button(root, text="Select Image", command=classify_image)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Upload an image to classify", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
