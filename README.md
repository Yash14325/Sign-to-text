# Sign Language To Text and Speech Conversion

This application converts sign language gestures captured through a webcam into text and speech. It uses a pre-trained machine learning model to recognize hand gestures and translate them into letters and words.

## Features

- Real-time sign language detection and recognition
- Text generation from recognized signs
- Word suggestions based on current text
- Text-to-speech functionality
- Responsive web interface
- Support for special gestures (space, backspace, etc.)

## Technologies Used

- **Backend**: Flask, TensorFlow, OpenCV, MediaPipe
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Other**: pyttsx3 for text-to-speech, pyenchant for spell checking

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Sign-Language-To-Text-and-Speech-Conversion-master
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Allow camera access when prompted by your browser.

4. Position your hand in the webcam view and make sign language gestures.

5. The application will recognize the gestures and convert them to text.

6. Use the suggestion buttons to select words if needed.

7. Click the "Speak" button to hear the text spoken aloud.

8. Click the "Clear" button to start a new sentence.

## Special Gestures

- **Space**: Special hand position to add a space
- **Backspace**: Special hand position to delete the last character
- **Next**: Special hand position to confirm a character

## Model Information

The application uses a pre-trained CNN model (`cnn8grps_rad1_model.h5`) for sign language recognition. The model is trained to recognize American Sign Language (ASL) gestures.

## Requirements

- Python 3.7+
- Webcam
- Modern web browser with JavaScript enabled

## License

This project is licensed under the MIT License - see the LICENSE file for details.