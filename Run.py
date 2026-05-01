import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
from PIL import Image, ImageTk
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
import os

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# App setup
app = ctk.CTk()
app.geometry("800x600")
app.title("Logo Detection")
app.configure(fg_color="#222222")

# SQLite setup
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT, password TEXT, phone TEXT, email TEXT,
        gender TEXT, address TEXT
    )
''')
conn.commit()

# Global user
current_user = None

# Function: Clear screen widgets
def clear_widgets():
    for widget in app.winfo_children():
        widget.destroy()

# Function: Register page
def register_page():
    clear_widgets()
    ctk.CTkLabel(app, text="User Registration", font=("Arial", 24)).pack(pady=20)
    
    global reg_username, reg_password, reg_phone, reg_email, reg_gender, reg_address
    reg_username = ctk.CTkEntry(app, placeholder_text="Username")
    reg_username.pack(pady=5)
    reg_password = ctk.CTkEntry(app, placeholder_text="Password", show="*")
    reg_password.pack(pady=5)
    reg_phone = ctk.CTkEntry(app, placeholder_text="Phone")
    reg_phone.pack(pady=5)
    reg_email = ctk.CTkEntry(app, placeholder_text="Email")
    reg_email.pack(pady=5)
    reg_gender = ctk.CTkOptionMenu(app, values=["Male", "Female", "Other"])
    reg_gender.pack(pady=5)
    reg_address = ctk.CTkTextbox(app, height=70, width=300)
    reg_address.pack(pady=5)
    
    ctk.CTkButton(app, text="Register", command=register_user).pack(pady=10)
    ctk.CTkButton(app, text="Back to Login", command=login_page).pack()

# Function: Login page
def login_page():
    clear_widgets()
    ctk.CTkLabel(app, text="Login", font=("Arial", 24)).pack(pady=20)

    global log_username, log_password
    log_username = ctk.CTkEntry(app, placeholder_text="Username")
    log_username.pack(pady=5)
    log_password = ctk.CTkEntry(app, placeholder_text="Password", show="*")
    log_password.pack(pady=5)

    ctk.CTkButton(app, text="Login", command=login_user).pack(pady=10)
    ctk.CTkButton(app, text="Register", command=register_page).pack()

# Function: Register user
def register_user():
    username = reg_username.get()
    password = reg_password.get()
    phone = reg_phone.get()
    email = reg_email.get()
    gender = reg_gender.get()
    address = reg_address.get("1.0", "end-1c")

    if not all([username, password, phone, email, gender, address]):
        messagebox.showerror("Error", "Please fill all fields")
        return

    cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)", (username, password, phone, email, gender, address))
    conn.commit()
    messagebox.showinfo("Success", "Registration successful!")
    login_page()

# Function: Login user
def login_user():
    global current_user
    username = log_username.get()
    password = log_password.get()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    if cursor.fetchone():
        current_user = username
        homepage()
    else:
        messagebox.showerror("Login Failed", "Invalid credentials")

# Function: Homepage
def homepage():
    clear_widgets()
    ctk.CTkLabel(app, text=f"Welcome {current_user}", font=("Arial", 22)).pack(pady=20)
    ctk.CTkButton(app, text="Upload Image to Detect Logo", command=browse_image).pack(pady=10)
    ctk.CTkButton(app, text="Logout", command=login_page).pack(pady=10)

# Function: Browse and detect
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        detect_logo(file_path)

# Function: Detect logo using Roboflow
def detect_logo(image_path):
    try:
        rf = Roboflow(api_key="GBXneGBgt2OeLyBC5BSJ")
        project = rf.workspace().project("logo-detector-cgxef")
        model = project.version(2).model

        image = cv2.imread(image_path)
        result = model.predict(image_path, confidence=40, overlap=30).json()

        boxes, confidences, class_names = [], [], []

        for pred in result["predictions"]:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            boxes.append([x1, y1, x2, y2])
            confidences.append(pred["confidence"])
            class_names.append(pred["class"])

        unique_classes = list(set(class_names))
        class_name_to_id = {name: idx for idx, name in enumerate(unique_classes)}
        class_ids = [class_name_to_id[name] for name in class_names]

        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )

        label_annotator = sv.LabelAnnotator()
        box_annotator = sv.BoxAnnotator()

        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=class_names)

        sv.plot_image(image=annotated_image, size=(12, 12))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to detect logo: {e}")

# Start app with login page
login_page()
app.mainloop()
