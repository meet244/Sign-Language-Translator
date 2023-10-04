# Sign Language Translator

### Translate sign language to text with camera and python (GUI and ML) 📷🤖📝

https://github.com/meet244/Sign-Language-Translator/assets/83262693/8020b9f0-6c23-4af4-a43c-bae3d59a8899

## Problem Statement 🧩

Develop an innovative system that utilizes camera technology in web and mobile applications to translate sign language gestures into text. The primary goal is to enhance communication accessibility for the Deaf and Hard of Hearing community by providing a real-time sign language-to-text translation solution. 🌐🤟📱

## Key Features 🚀

1. **Real-Time Gesture Recognition:** Advanced algorithms for recognizing sign language gestures in real-time through the device's camera. 📹👋

2. **Text Translation:** Accurate translation mechanism to convert recognized gestures into text. 📝🔄

3. **Accessible Interface:** User-friendly interface for both sign language users and those who rely on the translated text. 🖥️👨‍👩‍🦳

4. **Multiple Sign Languages:** Support for a variety of sign languages to cater to a diverse user base. 🌍🤟

5. **Customizable Settings:** Allow users to personalize the system's settings and preferences. ⚙️🛠️

## Solution Overview 🌟

We first understood how sign language functions and what the signs for India. Here's the signs we made this project for and you can try this out : 

![signs](https://github.com/meet244/Sign-Language-Translator/assets/83262693/30087850-85a3-4850-bdc4-fbe3daf87cc6)

We solved this problem by implementing a comprehensive system that combines customtkinter (an enhanced version of tkinter), Mediapipe for hand sign recognition, and TensorFlow for machine learning to recognize signs. Here's a more detailed breakdown of our solution:

- **Real-Time Gesture Recognition:** We leveraged Mediapipe's advanced hand tracking capabilities to recognize sign language gestures in real-time through the device's camera. This allowed us to precisely track hand movements and gestures. 👐🕐

- **Text Translation:** To convert recognized sign language gestures into text, we utilized TensorFlow for machine learning. Our machine learning model was trained to understand a wide range of sign language signs, ensuring high accuracy and reliability in translation. 🤖💬

- **Accessible Interface:** We designed a user-friendly interface using customtkinter, which offers enhanced customization and a smoother user experience. Our interface facilitates seamless communication for both sign language users and those who rely on the translated text. 🖼️🤝


## Installation ⚙️

```shell
# Clone the repository
git clone https://github.com/meet244/Sign-Language-Translator.git
cd Sign-Language-Translator

# Install modules
pip install -r requirements.txt

# Start the application
python app.py
```

## Contributing 🤝

If you'd like to contribute to this project, please follow guidelines. 🙌

## License 📜

This project is licensed under the [MIT License](LICENSE). 📄

