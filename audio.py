import speech_recognition as sr
import pyttsx3

#  ________________________________________________
# TEXT TO SPEECH SETUP
#  ________________________________________________
engine = pyttsx3.init()

def speak(text):
    """Convert text to speech safely."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS Error:", e)


#  ________________________________________________
# LISTEN FUNCTION 
# ________________________________________________
def listen(device_index=None):
    """
    Listen from microphone and return recognized text.
    """

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(device_index=device_index) as source:
            print("Listening...")
            # Adjust for background noise before capturing
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Capture audio with timeout and phrase limit
            audio_data = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

        # Use Google Speech Recognition API
        text = recognizer.recognize_google(audio_data)
        print(f"You said: {text}")
        return text.lower()

    except sr.WaitTimeoutError:
        print("Listening timed out.")
        return ""

    except sr.UnknownValueError:
        print("Couldnt understand audio.")
        return ""

    except sr.RequestError:
        print("Speech recognition service unavailable.")
        return ""

    except OSError as e:
        print("Microphone error:", e)
        return ""


#  ________________________________________________
# MATCH VOICE TO DETECTED OBJECTS
#  ________________________________________________
def find_item_in_screen(text, detected_objects):
    """
    Check if spoken text matches any detected objects.
    detected_objects: list of strings
    """

    for obj in detected_objects:
        if obj.lower() in text:
            speak("Item found")
            print("Item found")
            return True

    speak("Item not found")
    print("Item not found")
    return False
