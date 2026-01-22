# Gesture-Controlled Virtual Keyboard + Gemini AI (OpenCV + MediaPipe)

A real-time **computer vision based text input system** that lets you type **hands-free** using finger gestures.  
This project uses **OpenCV + Google MediaPipe Hands** to track landmarks from both hands, maps finger geometry to keyboard characters, and sends the final prompt to the **Gemini API** to generate an AI response.

---

## 🚀 Features

- 🎥 Live webcam hand tracking using **MediaPipe Hands**
- ⌨️ **Angle-based character selection** using left hand thumb–index direction
- 🤏 Pinch gestures for typing + editing (multi-command)
- 📝 On-screen text box for live input preview
- 🤖 Sends prompt to **Gemini API** and displays response instantly
- 🧠 Built using real-time geometry + gesture mapping (no physical keyboard)

---

## ✋ Finger Tracking (7 Fingers Total)

### Left Hand (3 fingers)
- Thumb
- Index
- Middle

### Right Hand (4 fingers)
- Thumb
- Index
- Middle
- Ring

---

## 🧠 Gesture Controls

### 🎛️ Character Selection (Left Hand)
**Left Thumb + Left Index**  
- The **angle with the horizontal axis** formed by the line between thumb tip and index tip is mapped to a **character array**.
- Each angle range corresponds to one character (like a gesture dial).

### ✅ Actions (Pinch Gestures)

#### Right Hand
| Gesture | Action |
|--------|--------|
| Right Thumb + Index pinch | Type / select current character |
| Right Thumb + Middle pinch | Backspace / delete last character |
| Right Thumb + Ring pinch | Insert space |

#### Left Hand
| Gesture | Action |
|--------|--------|
| Left Thumb + Middle pinch | Submit prompt to Gemini API |

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** (real-time webcam processing + UI rendering)
- **Google MediaPipe Hands** (hand landmark detection)
- **Vector Math / Geometry**
  - angle calculation using `atan2`
  - Euclidean distance for pinch detection
- **Gemini API** (LLM response generation)

---

## ⚙️ How It Works (Pipeline)

1. Capture live video frames using OpenCV  
2. Run MediaPipe Hands to detect landmarks for both hands  
3. Extract fingertip coordinates for required fingers  
4. Compute:
   - **Angle** between left thumb-index → select character  
   - **Pinch distances** → detect actions  
5. Update text box string:
   - add character / delete / space  
6. Submit prompt using left-hand submit gesture  
7. Call Gemini API and display response in the UI  

---

## 📌 Real-World Applications

- ♿ **Assistive typing system** for accessibility & hands-free input  
- 🥽 **AR/VR typing interfaces** (no physical keyboard needed)  
- 🏥 **Touchless interaction** for hospitals/labs/clean rooms  
- 🤖 **Silent AI assistant prompting** in noisy or privacy-sensitive environments  
- 🚗 Gesture-based UI controls (with safe design improvements)

---

## 📂 Project Structure (Example)

```
.
├── main.py
├── requirements.txt
├── README.md
└── assets/
    ├── demo.gif
    └── screenshots/
```

---

## 🧩 Requirements

- Python 3.8+
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
opencv-python
mediapipe
numpy
google-generativeai
```

---

## 🔑 Gemini API Setup

1. Create a Gemini API key from Google AI Studio
2. Set it as an environment variable:

### Windows (PowerShell)
```powershell
setx GEMINI_API_KEY "YOUR_API_KEY"
```

### macOS/Linux
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

3. In code, load it like:

```python
import os
API_KEY = os.getenv("GEMINI_API_KEY")
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🧠 Notes / Improvements Ideas

- Add smoothing filters for stable angle selection
- Add cooldown/debounce timers for pinch actions
- Expand character set (uppercase, symbols, numbers)
- Add word prediction / autocomplete
- Improve UI using a proper GUI framework (Tkinter / PyQt / Web UI)

---

## 📸 Demo

(Add your GIF/video screenshot here)

Example:
- `assets/demo.gif`
- `assets/screenshots/`

---

## 🙌 Acknowledgements

- **Google MediaPipe** for hand landmark tracking
- **OpenCV** for real-time computer vision tools
- **Gemini API** for AI-powered prompt completion

---

## 📬 Contact

If you have suggestions or want to collaborate, feel free to connect with me on LinkedIn!

**Author:** Satyam Saman
