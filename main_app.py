import tkinter as tk
from tkinter import filedialog, ttk
import customtkinter as ctk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from model_functions import predict_disease, get_disease_severity
from utils import load_labels, setup_styles

# Set appearance mode and default color theme
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")


class PlantDiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Plant Disease Classifier")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Load disease labels
        self.labels = load_labels()

        # Initialize variables
        self.img_path = None
        self.original_img = None
        self.processed_img = None
        self.prediction_made = False

        # Setup UI colors and styles
        self.colors = setup_styles()

        # Create UI
        self.create_ui()

    def create_ui(self):
        # Main frame layout (2x2 grid)
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=9)

        # Header frame
        self.header_frame = ctk.CTkFrame(self, fg_color=self.colors["dark_green"])
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.header_frame.grid_columnconfigure(0, weight=1)

        # App title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Plant Disease Classification System",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="white"
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=10)

        # Left panel - Image Upload and Display
        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(0, weight=0)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_rowconfigure(2, weight=0)

        # Upload section
        self.upload_frame = ctk.CTkFrame(self.left_panel, fg_color=self.colors["light_green"])
        self.upload_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.upload_button = ctk.CTkButton(
            self.upload_frame,
            text="Upload Plant Image",
            command=self.upload_image,
            fg_color=self.colors["mid_green"],
            hover_color=self.colors["dark_green"],
            font=ctk.CTkFont(size=14)
        )
        self.upload_button.pack(pady=10, padx=20)

        # Image display area
        self.image_frame = ctk.CTkFrame(self.left_panel)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_frame, text="Upload an image to analyze", fg_color="gray90")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Analysis button
        self.analyze_frame = ctk.CTkFrame(self.left_panel)
        self.analyze_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        self.analyze_button = ctk.CTkButton(
            self.analyze_frame,
            text="Analyze Plant",
            command=self.analyze_plant,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_dark"],
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.analyze_button.pack(pady=10, fill=tk.X, padx=40)

        # Right panel - Results Display
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=5)
        self.right_panel.grid_rowconfigure(1, weight=5)

        # Disease prediction area
        self.disease_frame = ctk.CTkFrame(self.right_panel, fg_color=self.colors["light_beige"])
        self.disease_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.disease_frame.grid_columnconfigure(0, weight=1)

        self.disease_title = ctk.CTkLabel(
            self.disease_frame,
            text="Disease Prediction",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors["dark_green"]
        )
        self.disease_title.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.result_frame = ctk.CTkFrame(self.disease_frame)
        self.result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.disease_result = ctk.CTkLabel(
            self.result_frame,
            text="No prediction yet",
            font=ctk.CTkFont(size=16),
            justify="left"
        )
        self.disease_result.pack(padx=10, pady=10, anchor="w")

        self.confidence_label = ctk.CTkLabel(
            self.result_frame,
            text="Confidence: N/A",
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        self.confidence_label.pack(padx=10, pady=5, anchor="w")

        # Severity estimation area
        self.severity_frame = ctk.CTkFrame(self.right_panel, fg_color=self.colors["light_beige"])
        self.severity_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.severity_frame.grid_columnconfigure(0, weight=1)
        self.severity_frame.grid_rowconfigure(1, weight=1)

        self.severity_title = ctk.CTkLabel(
            self.severity_frame,
            text="Disease Severity Estimation",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors["dark_green"]
        )
        self.severity_title.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Placeholder for severity visualization
        self.severity_result_frame = ctk.CTkFrame(self.severity_frame)
        self.severity_result_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Loading indicator
        self.loading_label = ctk.CTkLabel(
            self,
            text="Processing...",
            fg_color=self.colors["dark_green"],
            text_color="white",
            corner_radius=10
        )
        # Don't pack this yet - will be shown during processing

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            self.img_path = file_path
            self.load_and_display_image(file_path)
            self.analyze_button.configure(state="normal")

            # Reset previous results
            self.disease_result.configure(text="No prediction yet")
            self.confidence_label.configure(text="Confidence: N/A")

            # Clear any previous severity visualization
            for widget in self.severity_result_frame.winfo_children():
                widget.destroy()

            self.prediction_made = False

    def load_and_display_image(self, file_path):
        # Load the image
        self.original_img = cv2.imread(file_path)

        # Display image in the UI
        display_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        display_img = self.resize_image_aspect_ratio(display_img, 400)

        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Remove previous content
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Create new label with image
        img_display = ctk.CTkLabel(self.image_frame, text="", image=img_tk)
        img_display.image = img_tk  # Keep reference
        img_display.grid(row=0, column=0, sticky="nsew")

    def resize_image_aspect_ratio(self, image, max_size):
        height, width = image.shape[:2]

        # Calculate aspect ratio
        aspect = width / height

        if height > width:
            new_height = max_size
            new_width = int(aspect * new_height)
        else:
            new_width = max_size
            new_height = int(new_width / aspect)

        return cv2.resize(image, (new_width, new_height))

    def analyze_plant(self):
        if not self.img_path or self.prediction_made:
            return

        # Show loading indicator
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.update()

        # Run analysis in separate thread to avoid UI freezing
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            # Run disease prediction
            disease_class, confidence, top_classes = predict_disease(self.original_img)
            disease_name = self.labels[disease_class]

            # Run severity estimation
            severity_percentage, severity_img = get_disease_severity(self.original_img)

            # Update UI with results (from main thread)
            self.after(0, lambda: self._update_disease_results(disease_name, confidence, top_classes))
            self.after(0, lambda: self._update_severity_results(severity_percentage, severity_img))

            self.prediction_made = True
        except Exception as e:
            self.after(0, lambda: self._show_error(str(e)))
        finally:
            # Hide loading indicator
            self.after(0, lambda: self.loading_label.place_forget())

    def _update_disease_results(self, disease_name, confidence, top_classes):
        # Format disease name (replace underscores with spaces, title case)
        formatted_name = disease_name.replace('_', ' ').title()

        # Update disease prediction display
        self.disease_result.configure(
            text=f"Detected Disease: {formatted_name}"
        )
        self.confidence_label.configure(
            text=f"Confidence: {confidence:.2f}%"
        )



    def _update_severity_results(self, severity_percentage, severity_img):
        # Clear previous content
        for widget in self.severity_result_frame.winfo_children():
            widget.destroy()

        # Create frame for text results
        text_frame = ctk.CTkFrame(self.severity_result_frame)
        text_frame.pack(fill="x", padx=10, pady=5, anchor="n")

        # Severity percentage text
        severity_text = ctk.CTkLabel(
            text_frame,
            text=f"Estimated Severity: {severity_percentage:.2f}%",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        severity_text.pack(pady=5, anchor="w")

        # Severity level categorization
        level_text = "Low"
        level_color = "green"
        if severity_percentage > 45:
            level_text = "High"
            level_color = "red"
        elif severity_percentage > 20:
            level_text = "Medium"
            level_color = "orange"

        severity_level = ctk.CTkLabel(
            text_frame,
            text=f"Severity Level: {level_text}",
            font=ctk.CTkFont(size=14),
            text_color=level_color
        )
        severity_level.pack(pady=5, anchor="w")

        # Create progress bar for severity
        progress_frame = ctk.CTkFrame(self.severity_result_frame)
        progress_frame.pack(fill="x", padx=20, pady=10)

        progress = ctk.CTkProgressBar(progress_frame)
        progress.pack(fill="x", pady=5)
        progress.set(severity_percentage / 100)

        # Create visualization frame
        viz_frame = ctk.CTkFrame(self.severity_result_frame)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Convert matplotlib figure to Tkinter widget
        fig = plt.figure(figsize=(6, 3))

        # Original image
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original")
        ax1.axis('off')

        # Masked image
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(severity_img, cmap='gray')
        ax2.set_title("Infected Region")
        ax2.axis('off')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _show_error(self, error_message):
        for widget in self.severity_result_frame.winfo_children():
            widget.destroy()

        error_label = ctk.CTkLabel(
            self.severity_result_frame,
            text=f"Error during analysis: {error_message}",
            text_color="red"
        )
        error_label.pack(padx=20, pady=20)


if __name__ == "__main__":
    app = PlantDiseaseApp()
    app.mainloop()