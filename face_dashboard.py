import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2
import face_recognition
import pickle
import numpy as np
import time
import datetime
from tkinter import ttk
import csv
import os
import pyttsx3
import threading

engine = pyttsx3.init()
spoken_names = set()

class FaceDashboard:

    def __init__(self, root):

        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)

        self.known_encodings = data["encodings"]
        self.known_names = data["names"]

        self.root = root
        self.root.title("AI Face Recognition System")
        self.root.geometry("1100x850")
        self.root.configure(bg="#1e1e2e")

        self.camera_running = False
        self.latest_frame = None
        self.recognition_running = False
        self.current_name = "---"
        self.current_confidence = "---"

        title = tk.Label(
            root,
            text="🤖 AI FACE RECOGNITION SYSTEM",
            font=("Segoe UI", 24, "bold"),
            bg="#1e1e2e",
            fg="#00ffcc"
        )
        
        title.pack(pady=10) 

        # Video frame
        video_frame = tk.Frame(
        root,
        bg="#1e1e2e"
        )
        video_frame.pack()
 
        self.video_label = tk.Label(
        video_frame,
        bg="#1e1e2e"
        )
        self.video_label.pack()

        # Info frame
        info_frame = tk.Frame(
            root,
            bg="#2a2a40",
            bd=2,
            relief="ridge"
        )

        info_frame.pack(pady=10)

        self.info_label = tk.Label(
        info_frame,
        text="Camera Stopped",
        bg="#2a2a40",
        fg="#ffcc00",
        font=("Segoe UI", 12, "bold")
        )
        self.info_label.pack()

        self.status_label = tk.Label(
            info_frame,
            text="🟢 System Online",
            font=("Segoe UI", 12, "bold"),
            bg="#2a2a40",
            fg="#00ff00"
        )
        self.status_label.pack()

        self.name_label = tk.Label(
            info_frame,
            text="Name: ---",
            font=("Segoe UI", 12, "bold"),
            bg="#2a2a40",
            fg="white"
        )
        self.name_label.pack()

        self.conf_label = tk.Label(
        info_frame,
        text="Confidence: ---",
        bg="#2a2a40",
        fg="white",
        font=("Segoe UI", 12, "bold")
        )
        self.conf_label.pack()

        self.face_count_label = tk.Label(info_frame, 
            text="Faces Detected: 0",
            bg="#2a2a40",
            fg="white",
            font=("Segoe UI", 12))
        self.face_count_label.pack()

        self.fps_label = tk.Label(info_frame, text="FPS: 0",
            bg="#2a2a40",
            fg="white",
            font=("Segoe UI", 12))
        self.fps_label.pack()

        self.time_label = tk.Label(info_frame, text="Time: --",
            bg="#2a2a40",
            fg="white",
            font=("Segoe UI", 12))
        self.time_label.pack()

        self.people_label = tk.Label(
        info_frame,
        text=f"Known People: {len(set(self.known_names))}",
        bg="#2a2a40",
        fg="#00ffcc",
        font=("Segoe UI", 12, "bold")
        )
        self.people_label.pack()

        self.prev_time = time.time()

        # Frame counter (for smoother camera)
        self.frame_count = 0

        self.logged_today = set()
        self.attendance_file = "attendance.csv"

        if not os.path.exists(self.attendance_file):

            with open(self.attendance_file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                "Name",
                "Date",
                "Time",
                "Confidence"
                ])

        # Button frame
        button_frame = tk.Frame(
        root,
        bg="#1e1e2e"
        )
        button_frame.pack(pady=10)

        tk.Button(
        button_frame,
        text="Start Camera",
        command=self.start_camera,
        width=15,
        bg="#00b894",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        cursor="hand2"
        ).grid(row=0, column=0, padx=10)

        tk.Button(
        button_frame,
        text="Stop Camera",
        command=self.stop_camera,
        width=15,
        bg="#e17055",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        cursor="hand2"
        ).grid(row=0, column=1, padx=10)

        tk.Button(
        button_frame,
        text="View Attendance",
        command=self.view_attendance,
        width=15,
        bg="#0984e3",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        cursor="hand2",
        ).grid(row=0, column=3, padx=10)

        tk.Button(
        button_frame,
        text="Exit",
        command=self.close_app,
        width=15,
        bg="#d63031",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        cursor="hand2"
        ).grid(row=0, column=2, padx=10)

        tk.Button(
        button_frame,
        text="Add Person",
        command=self.add_person,
        width=15,
        bg="#6c5ce7",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        cursor="hand2"
        ).grid(row=0, column=4, padx=10)

        footer = tk.Label(
        root,
        text="Developed by Suhaib Ahmed • AI Smart Attendance System",
        bg="#1e1e2e",
        fg="#888888",
        font=("Segoe UI", 9)
        )
        footer.pack(pady=10)

    def start_camera(self):

        if self.camera_running:
            return

        self.camera_running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.update_frame()

    def update_frame(self):

        if not self.camera_running:
            return

        success, frame = self.cap.read()
        self.frame_count += 1

        if success:

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

            if self.frame_count % 3 == 0:

                locations = face_recognition.face_locations(
                small,
                model="hog"
            )

                encodings = face_recognition.face_encodings(
                small,
                locations
            )

            self.face_count_label.config(
            text=f"Faces Detected: {len(locations)}"
            )

            if len(locations) == 0:
                self.name_label.config(text="Name: ---")
                self.conf_label.config(text="Confidence: ---")

            for encoding in encodings:
                
                self.name_label.config(text="Name: Unknown")
                self.conf_label.config(text="Confidence: --")
                
                matches = face_recognition.compare_faces(
                self.known_encodings,
                encoding,
                tolerance=0.55
                )

                distances = face_recognition.face_distance(
                self.known_encodings,
                encoding
                )
                
                if len(distances) > 0:

                    best = np.argmin(distances)

                    if matches[best]:

                        confidence = (
                        1 - distances[best]
                        ) * 100

                        self.name_label.config(
                        text=f"Name: {self.known_names[best].upper()}"
                        )

                        self.conf_label.config(
                        text=f"Confidence: {confidence:.0f}%"
                        )

                        self.log_attendance(
                        self.known_names[best].upper(),
                        confidence
                        )

                        if self.known_names[best].upper() not in spoken_names:

                            engine.say(f"Welcome {self.known_names[best]}")
                            engine.runAndWait()

                            spoken_names.add(
                            self.known_names[best].upper()
                        )

            current_time = time.time()

            fps = 1 / (current_time - self.prev_time)

            self.prev_time = current_time

            self.fps_label.config(
            text=f"FPS: {int(fps)}"
            )

            current_clock = datetime.datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
            )

            self.time_label.config(
                text=f"Time: {current_clock}"
                )            

            img = Image.fromarray(rgb)
            img = img.resize((800, 450))

            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def stop_camera(self):

        self.camera_running = False

        if hasattr(self, "cap"):
            self.cap.release()

        self.video_label.configure(image="")
        self.info_label.config(text="Camera Stopped")

    def log_attendance(self, name, confidence):

        if name in self.logged_today:
            return

        now = datetime.datetime.now()

        with open(self.attendance_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
            name,
            now.strftime("%d-%m-%Y"),
            now.strftime("%H:%M:%S"),
            f"{confidence:.0f}%"
            ])

        self.logged_today.add(name)

        print(f"Attendance logged: {name}")

    def view_attendance(self):
    
        window = tk.Toplevel(self.root)
        window.configure(bg="#1e1e2e")
        window.title("Attendance Records")
        window.geometry("700x400")

        tree = ttk.Treeview(
        window,
        columns=("Name", "Date", "Time", "Confidence"),
        show="headings"
        )
        

        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")
        tree.heading("Confidence", text="Confidence")

        tree.pack(fill="both", expand=True)

        try:
            with open("attendance.csv", "r") as file:

                reader = csv.reader(file)

                next(reader)  # skip header

                for row in reader:
                        tree.insert("", "end", values=row)

        except FileNotFoundError:
            pass


    def close_app(self):

        self.stop_camera()
        self.root.destroy()
    
    def add_person(self):

        self.stop_camera()

        name = simpledialog.askstring(
        "Add Person",
        "Enter Person Name:"
        )

        if not name:
            self.start_camera()
            return

        folder = os.path.join(
        "known_faces",
        name.upper()
        )

        os.makedirs(folder, exist_ok=True)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("ERROR: Could not open camera")
            self.start_camera()
            return

        count = 0

        while count < 5:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"Capturing Image {count+1}/5",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Capture New Person", frame)

            cv2.waitKey(700)

            filename = os.path.join(
            folder,
            f"{count+1}.jpg"
            )

            cv2.imwrite(filename, frame)

            count += 1

        cap.release()
        cv2.destroyAllWindows()

        print("Updating face encodings...")

        import subprocess

        subprocess.run(["python", "encode_faces.py"])

        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)

        self.known_encodings = data["encodings"]
        self.known_names = data["names"]

        print("Known people loaded:")
        print(sorted(set(self.known_names)))

        self.start_camera()

        print(f"{name} added successfully!")

root = tk.Tk()
app = FaceDashboard(root)
root.mainloop()