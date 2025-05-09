# Plant Disease Classification App

A user-friendly GUI application for classifying plant diseases and estimating their severity using deep learning.

## Features

- Plant disease classification using a pre-trained CNN model
- Disease severity estimation
- Clean, intuitive, and visually appealing interface
- Real-time processing of uploaded images
- Visualization of infected regions
- Confidence scores and top disease predictions

## Screenshots

![Application Screenshot](https://via.placeholder.com/800x600?text=Plant+Disease+App+Screenshot)

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Tensorflow 2.x
- Your trained model file (`model_epoch_20.keras`)

### Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your trained model file (`model_epoch_20.keras`) is in the root directory of the application.

### Running the Application

Simply run the main script:

```bash
python main_app.py
```

## How to Use

1. Launch the application
2. Click "Upload Plant Image" to select a plant leaf image
3. After selecting an image, click "Analyze Plant"
4. The application will process the image and display:
   - The predicted disease class
   - Confidence level for the prediction
   - Top alternative predictions
   - Disease severity estimation
   - Visualization of the infected regions

## Project Structure

- `main_app.py`: Main application with Tkinter GUI
- `model_functions.py`: Model loading, prediction, and severity analysis functions
- `utils.py`: Utility functions for labels and styling
- `model_epoch_20.keras`: Your trained model file (not included in repo)

## Custom Labels

If you want to use custom labels, create a `labels.json` file in the root directory with a mapping of class indices to label names:

```json
{
  "0": "Apple_Scab",
  "1": "Apple_Black_Rot",
  ...
}
```

## Customization

You can customize the appearance of the application by modifying the color scheme in the `setup_styles()` function in `utils.py`.

## Troubleshooting

- **Model not found**: Ensure that `model_epoch_20.keras` is in the same directory as the application
- **Dependencies issues**: Make sure all required packages are installed using `pip install -r requirements.txt`
- **Display issues**: The application requires a GUI environment. If running on a server, ensure X11 forwarding is enabled

