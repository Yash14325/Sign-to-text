// Main JavaScript for Sign Language Conversion App

// Global variables
let video;
let isProcessing = false;
let processingInterval;
let currentSuggestions = ["", "", "", ""];
let cameraStream = null;
let isCameraRunning = true;
let processingSpeed = 10; // milliseconds between frame processing (lower = faster, reduced for better synchronization)

// DOM elements
const webcamElement = document.getElementById('webcam');
const processedImageElement = document.getElementById('processed-image');
const noHandMessageElement = document.getElementById('no-hand-message');
const currentCharacterElement = document.getElementById('current-character');
const sentenceElement = document.getElementById('sentence');
const suggestionsContainer = document.getElementById('suggestions-container');
const suggestionButtons = document.querySelectorAll('.suggestion-btn');
const speakButton = document.getElementById('speak-btn');
const clearButton = document.getElementById('clear-btn');
const loadingIndicator = document.getElementById('loading-indicator');

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeWebcam();
    setupEventListeners();
});

// Initialize webcam access
async function initializeWebcam() {
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamElement.srcObject = stream;
        cameraStream = stream;
        video = webcamElement;
        isCameraRunning = true;

        // Wait for video to be ready
        webcamElement.addEventListener('loadeddata', () => {
            // Start processing frames
            startProcessing();
        });
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Error accessing webcam. Please ensure you have granted camera permissions.');
    }
}

// Stop the webcam
function stopCamera() {
    if (cameraStream) {
        const tracks = cameraStream.getTracks();
        tracks.forEach(track => track.stop());
        webcamElement.srcObject = null;
        isCameraRunning = false;
        
        if (processingInterval) {
            clearInterval(processingInterval);
            processingInterval = null;
        }
    }
}

// Start the webcam
async function startCamera() {
    if (!isCameraRunning) {
        try {
            await initializeWebcam();
        } catch (error) {
            console.error('Error restarting camera:', error);
        }
    }
}

// Set up event listeners
function setupEventListeners() {
    // Suggestion buttons
    suggestionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const index = button.getAttribute('data-index');
            const word = currentSuggestions[index];
            if (word && word.trim() !== '') {
                selectWord(word);
            }
        });
    });

    // Speak button
    speakButton.addEventListener('click', () => {
        const text = sentenceElement.textContent;
        if (text && text.trim() !== '') {
            speakText(text);
        }
    });

    // Clear button
    clearButton.addEventListener('click', clearText);
    
    // Camera toggle button
    const cameraToggleBtn = document.getElementById('camera-toggle');
    const cameraToggleText = document.getElementById('camera-toggle-text');
    
    cameraToggleBtn.addEventListener('click', () => {
        if (isCameraRunning) {
            stopCamera();
            cameraToggleText.textContent = 'Start Camera';
            cameraToggleBtn.querySelector('i').classList.remove('fa-video');
            cameraToggleBtn.querySelector('i').classList.add('fa-video-slash');
        } else {
            startCamera();
            cameraToggleText.textContent = 'Stop Camera';
            cameraToggleBtn.querySelector('i').classList.remove('fa-video-slash');
            cameraToggleBtn.querySelector('i').classList.add('fa-video');
        }
    });
}

// Start processing frames
function startProcessing() {
    if (processingInterval) {
        clearInterval(processingInterval);
    }

    // Process frames at the specified processing speed
    processingInterval = setInterval(() => {
        if (!isProcessing && video && video.readyState === 4 && isCameraRunning) {
            processFrame();
        }
    }, processingSpeed);
}

// Process a single frame
async function processFrame() {
    isProcessing = true;

    try {
        // Create a canvas to capture the current frame
        const canvas = document.createElement('canvas');
        // Use smaller dimensions for faster processing and better synchronization
        canvas.width = 320; // Reduced size for faster processing
        canvas.height = 240; // Reduced size for faster processing
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the frame to base64 with reduced quality for faster transmission
        const imageData = canvas.toDataURL('image/jpeg', 0.7); // Further reduced quality for faster transmission

        // Send the frame to the server for processing
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        // Update the UI with the results
        updateUI(result);
    } catch (error) {
        console.error('Error processing frame:', error);
    } finally {
        isProcessing = false;
    }
}

// Update the UI with the processing results
function updateUI(result) {
    if (result.error) {
        console.error('Error from server:', result.error);
        return;
    }

    // Update processed image if available
    if (result.processed_image) {
        processedImageElement.src = result.processed_image;
        processedImageElement.style.display = 'block';
        noHandMessageElement.style.display = 'none';
    } else {
        processedImageElement.style.display = 'none';
        noHandMessageElement.style.display = 'block';
    }

    // Update current character
    if (result.character) {
        // Don't display special characters like 'next', 'Backspace', etc.
        if (['next', 'Backspace', '  '].includes(result.character)) {
            currentCharacterElement.textContent = result.character;
            currentCharacterElement.classList.remove('highlight-character');
        } else {
            currentCharacterElement.textContent = result.character;
            // Highlight the current character to make it more visible
            currentCharacterElement.classList.add('highlight-character');
        }
    } else {
        currentCharacterElement.textContent = '-';
        currentCharacterElement.classList.remove('highlight-character');
    }

    // Update sentence
    if (result.sentence) {
        sentenceElement.textContent = result.sentence;
    }

    // Update suggestions
    if (result.suggestions && Array.isArray(result.suggestions)) {
        currentSuggestions = result.suggestions;
        suggestionButtons.forEach((button, index) => {
            const suggestion = result.suggestions[index] || '-';
            button.textContent = suggestion;
            button.disabled = suggestion === '-' || suggestion.trim() === '';
        });
    }
}

// Select a suggested word
async function selectWord(word) {
    try {
        const response = await fetch('/select_word', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ word })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success && result.sentence) {
            sentenceElement.textContent = result.sentence;
        }
    } catch (error) {
        console.error('Error selecting word:', error);
    }
}

// Speak the current text
async function speakText(text) {
    try {
        const response = await fetch('/speak', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Speech result:', result);

        // For browsers that support the Web Speech API, we can also speak in the browser
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    } catch (error) {
        console.error('Error speaking text:', error);
    }
}

// Clear the current text
async function clearText() {
    try {
        const response = await fetch('/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            sentenceElement.textContent = '';
            currentCharacterElement.textContent = '-';
            suggestionButtons.forEach(button => {
                button.textContent = '-';
                button.disabled = true;
            });
            currentSuggestions = ["", "", "", ""];
        }
    } catch (error) {
        console.error('Error clearing text:', error);
    }
}

// Handle visibility change to pause/resume processing
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause processing
        if (processingInterval) {
            clearInterval(processingInterval);
            processingInterval = null;
        }
    } else {
        // Page is visible again, resume processing
        if (!processingInterval) {
            startProcessing();
        }
    }
});