import pyttsx3
 
engine = pyttsx3.init()
 
def speak(message):
    engine.setProperty('rate',192)
    engine.say(message)
    engine.runAndWait()