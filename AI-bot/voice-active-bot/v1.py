from gtts import gTTS   
import speech_recognition as sr 
import os 
import webbrowser 
import smtplib 
import winsound


def strt(audo):
    print(audo)
    obj1 = gTTS(text = audo,lang = 'en',slow = False)
    obj1.save("audio.mp3")
    winsound.PlaySound("audio.mp3",winsound.SND_ASYNC)


def commandenter():
    #speeh recog
    # command = None
    r = sr.Recognizer()
    with sr.Microphone() as srcvar:
        print(" Start Speaking \n")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(srcvar,duration = 1)
        audio = r.listen(srcvar)
        print("Reached")

    try:
        command = r.recognize_google(audio)
        print("Touche "+command+" \n")

    except sr.UnknownValueError:
        print("Sorry not understood \n")
        mainassist_fucn(commandenter())
        command = "Not Understood"

    return command

def mainassist_fucn(commda):
    chom_mepath = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    print(commda)
    if 'open Reddit python' in commda:
        print("reached")
        urlvar = "https://www.reddit.com/r/Python/"
        webbrowser.open_new(urlvar)
        pass
    #create cases may be audio processing tokenize ?
    if 'open youtube' in commda:
        print("reached")
        urlvar = "https://www.reddit.com/r/Python/"
        webbrowser.open_new(urlvar)
        pass
    if "quit" in commda:
        return 7
# mainassist_fucn("open Reddit python")


while True: 
    if mainassist_fucn(commandenter())==7:
        break
    pass


