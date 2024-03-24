import math
from ultralytics import YOLO
import cv2
import cvzone
import easyocr
from openpyxl import Workbook
from openpyxl.drawing.image import Image as xlImage
from openpyxl.styles import Alignment
import tempfile
import os
import xlwings as xw
import tkinter as tk
from tkinter import filedialog
from difflib import SequenceMatcher

model = YOLO("../Yolo-Weights/yolov8n.pt")
plate_cascade = cv2.CascadeClassifier("model/haarcascade_russian_plate_number.xml")
reader = easyocr.Reader(['en'])

CAR_CLASS_INDEX = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789-"

def track_and_extract_plate(img, box):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    plate_region = img[y1:y2, x1:x2]
    return plate_region

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def perform_ocr_on_plate(plate_img, existing_texts):
    results = reader.readtext(plate_img)
    if results:
        number_plate_text = results[0][1]
        easyocr_accuracy = results[0][2]  # Get the confidence score from EasyOCR
        similarity_rate = calculate_similarity(number_plate_text, existing_texts)  # Calculate similarity with previous texts
        return number_plate_text, easyocr_accuracy, similarity_rate
    else:
        return None, None, None

def calculate_similarity(current_plate_text, existing_texts):
    # Calculate similarity with previously detected plate texts
    if existing_texts:
        max_similarity = 0
        for prev_plate_text in existing_texts:
            similarity = similar(current_plate_text, prev_plate_text)
            if similarity > max_similarity:
                max_similarity = similarity
        return max_similarity
    else:
        return None

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    wb = xw.Book()
    sht = wb.sheets[0]

    sht.cells(1, 'A').value = "Images"
    sht.cells(1, 'B').value = "Number Plate Text"
    sht.cells(1, 'C').value = "EasyOCR Accuracy"
    sht.cells(1, 'D').value = "Similarity Rate"

    current_row = 2
    existing_texts = set()

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == CAR_CLASS_INDEX:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    plate_img = track_and_extract_plate(img, (x1, y1, x2, y2))

                    plates = plate_cascade.detectMultiScale(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
                    for (px, py, pw, ph) in plates:
                        plate_text, easyocr_accuracy, similarity_rate = perform_ocr_on_plate(plate_img[py:py+ph, px:px+pw], existing_texts)
                        if plate_text and all(char in ALPHABET for char in plate_text) and plate_text not in existing_texts:
                            print("License Plate Text:", plate_text)

                            if similarity_rate is not None and similarity_rate > 0.6:
                                print("Similarity rate is high. Skipping insertion into Excel.")
                                continue

                            cell = sht.cells(current_row, 'A')
                            left = cell.left
                            top = cell.top
                            width = cell.width
                            height = cell.height
                            plate_img_resized = cv2.resize(plate_img[py:py+ph, px:px+pw], (int(width), int(height)))

                            temp_img_path = os.path.join(tempfile.gettempdir(), 'plate_image.png')
                            cv2.imwrite(temp_img_path, plate_img_resized)

                            sht.pictures.add(temp_img_path, name=f'picture_{current_row}', left=left, top=top, width=width, height=height)

                            sht.cells(current_row, 'B').value = plate_text
                            sht.cells(current_row, 'B').api.HorizontalAlignment = -4108

                            sht.cells(current_row, 'C').value = easyocr_accuracy
                            
                            # Handle None value for similarity rate
                            if similarity_rate is not None:
                                sht.cells(current_row, 'D').value = similarity_rate
                            else:
                                sht.cells(current_row, 'D').value = "N/A"

                            existing_texts.add(plate_text)
                            current_row += 1

                    cvzone.cornerRect(img, (x1, y1, w, h))

        cv2.imshow('Cars Recognized by YOLO', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    wb.save('plate_recognition_data.xlsx')

def upload_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if video_path:
        process_video(video_path)

def connect_to_camera():
    # Add your camera connection code here
    pass

def main():
    root = tk.Tk()
    root.title("License Plate Recognition App")
    root.geometry("400x200")

    upload_button = tk.Button(root, text="Upload Video", command=upload_video)
    upload_button.pack(pady=10)

    camera_button = tk.Button(root, text="Connect to Camera", command=connect_to_camera)
    camera_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
