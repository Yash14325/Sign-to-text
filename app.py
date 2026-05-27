# Importing Libraries
import numpy as np
import math
import cv2
import mediapipe as mp
import os, sys
import traceback
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import base64
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import json

# Initialize the Flask application
app = Flask(__name__, static_folder='static')

# Register the signs folder as a static folder
app.static_folder = 'static'
app.add_url_rule('/static/signs/<path:filename>', endpoint='signs', view_func=app.send_static_file)

# Initialize hand detectors with optimized parameters
hd = HandDetector(maxHands=1, detectionCon=0.6)
hd2 = HandDetector(maxHands=1, detectionCon=0.6)

# Initialize dictionary for spell checking
try:
    ddd = enchant.Dict("en-US")
except enchant.errors.DictNotFoundError:
    try:
        ddd = enchant.Dict("en_US")
    except enchant.errors.DictNotFoundError:
        ddd = None
except Exception:
    ddd = None

# Set offset for hand detection
offset = 29

# Set environment variables
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Load the model
model = load_model('cnn8grps_rad1_model.h5')

# Initialize text-to-speech engine
speak_engine = pyttsx3.init()
speak_engine.setProperty("rate", 100)
voices = speak_engine.getProperty("voices")
speak_engine.setProperty("voice", voices[0].id)

# Global variables to track state
ct = {}
ct['blank'] = 0
blank_flag = 0
space_flag = False
next_flag = True
prev_char = ""
count = -1
ten_prev_char = [" "] * 10

# Initialize character counters
for i in ascii_uppercase:
    ct[i] = 0

# Current state variables
current_symbol = "C"
str_text = " "
word = " "
word1 = " "
word2 = " "
word3 = " "
word4 = " "

# Helper function to calculate distance between two points
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def get_first_hand(detection_result):
    """Return the first hand dict from either cvzone return format."""
    if not detection_result:
        return None

    hands = detection_result
    if isinstance(detection_result, tuple):
        hands = detection_result[0]

    if isinstance(hands, dict):
        return hands

    if isinstance(hands, (list, tuple)) and hands:
        first = hands[0]
        if isinstance(first, dict):
            return first
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], dict):
            return first[0]

    return None

# Function to check if palm is shown (to add character to sentence)
def check_palm_gesture(pts):
    # Check if all fingers are extended (palm is open and facing camera)
    is_palm = (pts[4][1] < pts[3][1] and  # Thumb is up
              pts[8][1] < pts[6][1] and  # Index finger is extended
              pts[12][1] < pts[10][1] and  # Middle finger is extended
              pts[16][1] < pts[14][1] and  # Ring finger is extended
              pts[20][1] < pts[18][1])  # Pinky is extended
    return is_palm

# Function to process image and predict sign
def predict_sign(image_data):
    global prev_char, current_symbol, count, ten_prev_char, str_text, word, word1, word2, word3, word4
    
    try:
        # Decode base64 image
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize for faster processing
        frame = cv2.resize(frame, (320, 240))
        
        # Flip the image horizontally for a later selfie-view display
        cv2image = cv2.flip(frame, 1)
        
        # Find hands in the image
        hands = hd.findHands(cv2image, draw=False, flipType=True)
        
        result = {
            "character": "",
            "sentence": str_text,
            "suggestions": [word1, word2, word3, word4]
        }
        
        hand_map = get_first_hand(hands)

        if hand_map:
            x, y, w, h = hand_map['bbox']

            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(cv2image.shape[1], x + w + offset)
            y2 = min(cv2image.shape[0], y + h + offset)
            
            # Extract hand region
            image = cv2image[y1:y2, x1:x2]
            
            # Load white background
            white = cv2.imread("white.jpg")
            if white is None:
                white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            elif white.shape[:2] != (400, 400):
                white = cv2.resize(white, (400, 400))
            
            if image.size > 0:
                handz = hd2.findHands(image, draw=False, flipType=True)
                
                hand_map = get_first_hand(handz)

                if hand_map:
                    pts = hand_map.get('lmList', [])
                    if len(pts) < 21:
                        return result
                    
                    # Calculate offsets for drawing
                    os_x = ((400 - w) // 2) - 15
                    os_y = ((400 - h) // 2) - 15
                    
                    # Draw hand landmarks and connections
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    
                    # Connect landmarks
                    cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), 
                            (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), 
                            (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), 
                            (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), 
                            (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), 
                            (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                    
                    # Draw circles at landmarks
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)
                    
                    # Process the image for prediction
                    ch1 = process_prediction(white, pts)
                    
                    # Check for palm gesture to add character to sentence
                    is_palm_shown = check_palm_gesture(pts)
                    
                    # Update state based on prediction
                    # The 'next' gesture now only confirms the current character
                    # We don't automatically add characters anymore, only when palm is shown

                    if ch1 == "  " and prev_char != "  ":
                        str_text = str_text + "  "

                    if ch1 == "Backspace" and prev_char != "Backspace":
                        str_text = str_text[0:-1]
                    
                    # Add a 'next' sign implementation
                    if ch1 == "next" and prev_char not in ["next", "  ", "Backspace"]:
                        # If previous character was a valid letter, append it to the sentence
                        str_text = str_text + prev_char
                        
                    # Only add character to sentence if palm is shown and it's a valid letter
                    elif is_palm_shown and ch1 not in ["next", "  ", "Backspace"] and prev_char != ch1:
                        str_text = str_text + ch1
                    
                    prev_char = ch1
                    current_symbol = ch1
                    count += 1
                    ten_prev_char[count%10] = ch1
                    
                    # Update word suggestions
                    if len(str_text.strip()) != 0:
                        st = str_text.rfind(" ")
                        ed = len(str_text)
                        word = str_text[st+1:ed]
                        
                        if len(word.strip()) != 0 and ddd is not None:
                            try:
                                suggestions = ddd.suggest(word)
                            except Exception:
                                suggestions = []
                            lenn = len(suggestions)
                            
                            word1 = suggestions[0] if lenn >= 1 else " "
                            word2 = suggestions[1] if lenn >= 2 else " "
                            word3 = suggestions[2] if lenn >= 3 else " "
                            word4 = suggestions[3] if lenn >= 4 else " "
                        else:
                            word1 = word2 = word3 = word4 = " "

                    
                    # Encode the processed image to send back
                    _, buffer = cv2.imencode('.jpg', white)
                    processed_img = base64.b64encode(buffer).decode('utf-8')
                    
                    # Update result with current state
                    result["character"] = current_symbol
                    result["sentence"] = str_text
                    result["suggestions"] = [word1, word2, word3, word4]
                    result["processed_image"] = f"data:image/jpeg;base64,{processed_img}"
        
        return result
    
    except Exception as e:
        print(f"Error in predict_sign: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

# Function to process prediction from the model
def process_prediction(white_img, pts):
    # Reshape image for model input
    white = white_img.reshape(1, 400, 400, 3)
    
    # Get model predictions
    prob = np.array(model.predict(white)[0], dtype='float32')
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    ch3 = np.argmax(prob, axis=0)
    prob[ch3] = 0

    pl = [ch1, ch2]
    
    # Apply various conditions to refine prediction
    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2
            
    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # Map numeric predictions to characters
    if isinstance(ch1, (int, np.integer)):
        if ch1 == 0:
            # Group 0 - [A, E, M, N, S, T]
            ch1 = 'S'
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                ch1 = 'A'
            if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                ch1 = 'T'
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                ch1 = 'E'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                ch1 = 'M'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                ch1 = 'N'
        elif ch1 == 1:
            # Group 1 - [B, D, F, I, K, R, U, V, W]
            ch1 = 'B'
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 'D'
        elif ch1 == 2:
            # Group 2 - [C, O]
            if distance(pts[11], pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'
        elif ch1 == 3:
            # Group 3 - [G, H]
            if (distance(pts[8], pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'
        elif ch1 == 4:
            # Group 4 - [L, next]
            # Check if this is the 'next' sign - specific hand configuration
            if pts[8][1] < pts[5][1] and pts[12][1] > pts[9][1] and pts[16][1] > pts[13][1] and pts[20][1] > pts[17][1]:
                ch1 = 'next'
            else:
                ch1 = 'L'
        elif ch1 == 5:
            # Group 5 - [P, Q, Z]
            if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                if pts[8][1] < pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'
        elif ch1 == 6:
            # Group 6 - [X]
            ch1 = 'X'
        elif ch1 == 7:
            # Group 7 - [Y, J]
            if distance(pts[8], pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'
    
    # Additional letter signs and refinements
    # Check for specific hand configurations that override the initial classification
    
    # F sign
    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
        ch1 = 'F'
    
    # I sign
    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
        ch1 = 'I'
    
    # W sign
    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
        ch1 = 'W'
    
    # K sign
    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
        ch1 = 'K'
    
    # U sign
    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
        ch1 = 'U'
    
    # V sign
    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
        ch1 = 'V'
    
    # R sign
    if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
        ch1 = 'R'
        
    # Ñ sign (if needed)
    if ch1 == 'N' and pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1] and pts[8][1] < pts[12][1]:
        ch1 = 'Ñ'
    
    # Check for special gestures
    # Space gesture
    if ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "  "
    
    # Next gesture
    if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
        if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "next"
    
    # Backspace gesture
    if ch1 in ['B', 'C', 'H', 'F', 'X']:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and \
           (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and \
           (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
            ch1 = 'Backspace'
    
    return ch1

# Route for the landing page (default home page)
@app.route('/')
def landing():
    return render_template('landing.html')

# Route for the detection page (former index page)
@app.route('/detection')
def detection():
    return render_template('index.html')

# Route to serve sign images
@app.route('/signs/<filename>')
def serve_sign(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'signs'), filename)

# Route for the learn signs page
@app.route('/learn')
def learn():
    # Get all sign images from the signs folder
    sign_images = []
    # Look for images in the root signs directory
    signs_folder = os.path.join(os.path.dirname(__file__), 'signs')
    for filename in sorted(os.listdir(signs_folder)):
        if filename.endswith('.png'):
            letter = filename.split('.')[0]  # Get the letter from the filename
            sign_images.append({
                'letter': letter
            })
    return render_template('learn.html', sign_images=sign_images)

# Route for the test yourself page
@app.route('/test')
def test():
    return render_template('test.html')

# API endpoint for processing webcam frames
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if request.method == 'POST':
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data provided"})
        
        result = predict_sign(image_data)
        return jsonify(result)

# API endpoint for text-to-speech
@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '')
    
    if text:
        # In a real application, we would use a proper TTS service
        # For now, we'll just return success
        return jsonify({"success": True, "message": f"Speaking: {text}"})
    
    return jsonify({"success": False, "message": "No text provided"})

# API endpoint for clearing text
@app.route('/clear', methods=['POST'])
def clear():
    global str_text, word1, word2, word3, word4
    str_text = " "
    word1 = word2 = word3 = word4 = " "
    return jsonify({"success": True, "message": "Text cleared"})

# API endpoint for selecting a word suggestion
@app.route('/select_word', methods=['POST'])
def select_word():
    global str_text
    
    data = request.get_json()
    selected_word = data.get('word', '')
    
    if selected_word and str_text:
        # Find the last word in the sentence and replace it
        idx_space = str_text.rfind(" ")
        if idx_space >= 0:
            str_text = str_text[:idx_space+1] + selected_word.upper()
        else:
            str_text = selected_word.upper()
        
        return jsonify({"success": True, "sentence": str_text})
    
    return jsonify({"success": False, "message": "Invalid word selection"})

# Run the application
if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
