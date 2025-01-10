import cv2
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
import threading

# Initialize the YOLO model
yolo = YOLO("Traffic_sign_detector.pt")

# Load the video capture
videoCap = cv2.VideoCapture(0)

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty(
    "rate",
    200,
)  # Set speech rate

# tts_engine.setProperty(
#     "voice",
#     "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0",
# )

# Create the main tkinter GUI
root = tk.Tk()
root.title("Driving Dashboard - Traffic Sign Detection")
root.geometry("1100x600")
root.configure(bg="black")

# Create a video feed section
video_frame = tk.Frame(root, bg="black", bd=5, relief=tk.RIDGE)
video_frame.place(x=20, y=20, width=640, height=480)
video_label = tk.Label(video_frame, bg="black")
video_label.pack()

# Create a detected traffic sign section
info_frame = tk.Frame(root, bg="black", bd=5, relief=tk.RIDGE)
info_frame.place(x=680, y=20, width=400, height=480)

info_label = tk.Label(
    info_frame,
    text="Detected Traffic Signs",
    font=("Helvetica", 16),
    fg="white",
    bg="black",
)
info_label.pack(anchor=tk.N, pady=10)

info_text = tk.Text(
    info_frame,
    height=50,
    width=55,
    font=("Helvetica", 28),
    fg="white",
    bg="black",
    relief=tk.FLAT,
)
info_text.pack()

# Footer section for dashboard-like appearance
footer = tk.Frame(root, bg="gray", bd=5, relief=tk.RIDGE)
footer.place(x=20, y=520, width=860, height=50)
footer_label = tk.Label(
    footer,
    text="Driving Dashboard Active",
    font=("Helvetica", 14),
    bg="gray",
    fg="white",
)
footer_label.pack()

voice_announced = set()


def announce_sign(class_name):
    tts_engine.say(f"{class_name}")
    tts_engine.runAndWait()


def update_frame():
    ret, frame = videoCap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Flip the frame horizontally to simulate a mirror effect
    # frame = cv2.flip(frame, 1)

    results = yolo.track(frame, stream=True)
    detected_signs = []

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                class_name = classes_names[cls]
                confidence = box.conf[0]

                # Add to detected signs list
                detected_signs.append(f"{class_name} ({confidence:.2f})")

                # Announce detected traffic signs (only once per sign)
                if class_name not in voice_announced:
                    threading.Thread(target=announce_sign, args=(class_name,)).start()
                    voice_announced.add(class_name)

                # Draw detection box and label
                colour = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    2,
                )

    # Update the detected signs in the text box
    info_text.delete("1.0", tk.END)
    info_text.insert(tk.END, "\n".join(detected_signs))

    # Convert the frame to an image format compatible with Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the function to run again
    root.after(10, update_frame)


def clear_voice_announced():
    voice_announced.clear()
    root.after(10000, clear_voice_announced)  # Schedule to run every 10 seconds


# Start updating the frames
update_frame()

# Start clearing the voice_announced set every 10 seconds
clear_voice_announced()


# Handle closing the app
def on_closing():
    videoCap.release()
    cv2.destroyAllWindows()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
