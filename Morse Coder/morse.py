from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
from threading import Event
import RPi.GPIO as GPIO
import re
import numpy as np
import time
import cv2

UNIT = 0.5
PIN = 13
FILE_PATH = 'input.txt'

"""
Morse code encoder using a LED on PIN on Raspberry Pi
Genenrates sequence based on UNIT seconds
"""

class MorseCode(object):

    code_table = {'A': '.-', 'B': '-...', 'C': '-.-.',
                  'D': '-..', 'E': '.', 'F': '..-.',
                  'G': '--.', 'H': '....', 'I': '..',
                  'J': '.---', 'K': '-.-', 'L': '.-..',
                  'M': '--', 'N': '-.', 'O': '---',
                  'P': '.--.', 'Q': '--.-', 'R': '.-.',
                  'S': '...', 'T': '-', 'U': '..-',
                  'V': '...-', 'W': '.--', 'X': '-..-',
                  'Y': '-.--', 'Z': '--..', '0': '-----',
                  '1': '.----', '2': '..---', '3': '...--',
                  '4': '....-',  '5': '.....', '6': '-....',
                  '7': '--...', '8': '---..', '9': '----.'}
				  
    def __init__(self, unit, pin):
        self.dot_t = 1*unit
        self.letter_t = 3*unit
        self.word_t = 7*unit
        self.int_lettter_t = 1*unit
        self.dash_t = 3*unit
        self.timing_table = {'.': self.dot_t, '-': self.dash_t,
                             ' ': self.letter_t}
        self.pin = pin
        self.GPIO_setup()

    def output_word(self, word):
        _word = word.upper()
        _m_word = ' '.join(MorseCode.code_table[letter] for letter in _word)
        for code in _m_word:
            if( code == ' '):
                GPIO.output(self.pin, False)
                time.sleep(self.timing_table[code])
            else:
                GPIO.output(self.pin, True)
                time.sleep(self.timing_table[code])
                GPIO.output(self.pin, False)
                time.sleep(self.int_lettter_t)
        time.sleep(self.word_t)
        return

    def GPIO_setup(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        print('GPIO setup ')
        GPIO.setup(self.pin, GPIO.OUT)

"""
Morse code decoder using camera on RPi
Decodes real time sequence (unfortunately hardwired)
Emprically tested due to syncronization of camera and LED flashing
"""
    
class VideoDecoder(MorseCode):
    
    def __init__(self, pin):
        self.pin = pin
        self.rev_code_table = {value:key for key,
                               value in MorseCode.code_table.items()}
        
    def calibrate(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        print('Calibrating')
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, False)
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        raw_capture = PiRGBArray(camera, size=(640, 480))
        time.sleep(0.5)
        camera.capture(raw_capture, format='bgr', use_video_port=True)
        background_im = cv2.cvtColor(raw_capture.array, cv2.COLOR_BGR2GRAY)
        raw_capture.truncate(0)
        GPIO.output(self.pin, True)
        camera.capture(raw_capture, format='bgr', use_video_port=True)
        time.sleep(2)
        led_im = cv2.cvtColor(raw_capture.array, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(led_im, background_im)
        ret_val, bin_im = cv2.threshold(frame_diff, 127, 255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(bin_im, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_area_index = np.argmax(areas)
        led_contour = contours[max_area_index]
        GPIO.output(self.pin, False)
        raw_capture.close()
        camera.close()
        x,y,w,h = cv2.boundingRect(led_contour)
        self.x1 = x
        self.y1 = y
        self.x2 = x+w
        self.y2 = y+h
    
    def video_capture(self):
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        raw_capture = PiRGBArray(camera, size=(640, 480))
        time.sleep(0.5)
        camera.capture(raw_capture, format='bgr', use_video_port=True)
        bg_im = cv2.cvtColor(raw_capture.array, cv2.COLOR_BGR2GRAY)
        led_bg_im = bg_im[self.y1:self.y2, self.x1:self.x2]
        raw_capture.truncate(0)
        on_count = 0
        off_count = 0
        letter = ''
        word = ''
        for frame in camera.capture_continuous(raw_capture,
                                               format='bgr', use_video_port=True):
            curr_im = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
            curr_im = curr_im[self.y1:self.y2, self.x1:self.x2]
            ret_val, bin_im = cv2.threshold(curr_im, 127, 255, cv2.THRESH_BINARY)
            bg_im[self.y1:self.y2, self.x1:self.x2] = bin_im
            if bin_im[bin_im > 0].size > 10 :
                on_count = on_count + 1
                if(off_count < 30 and off_count > 10):
                    word += self.rev_code_table[letter]
                    letter = ''
                if(off_count < 50 and off_count >= 30):
                    word += self.rev_code_table[letter]
                    print(word)
                    word = ''
##                print('off_count : ' + str(off_count))
                off_count = 0
            else:
                off_count = off_count + 1
                if(on_count < 10 and on_count > 0):
                    letter += '.'
                if(on_count < 20 and on_count > 10):
                    letter += '-'
##                print('on_count : ' + str(on_count))
                on_count = 0
            # display output
            cv2.imshow('frame', bg_im)
            key = cv2.waitKey(1) & 0xFF
            raw_capture.truncate(0)
            if key == ord('q'):
                word += self.rev_code_table[letter]
                print(word)
                break
                
        # close camera
        raw_capture.close()
        camera.close()
        cv2.destroyAllWindows()
        return


def morse_encode():
    m = MorseCode(UNIT, PIN)
    f = open(FILE_PATH)
    input_text = f.read()
    text = re.sub('[\r\n]+',' ',input_text)
    text = re.sub('[^a-zA-Z0-9 ]+','',text)
    time.sleep(10)
    for word in text.split():
        m.output_word(word)
    GPIO.cleanup()
    return
    
def morse_decode():
    v = VideoDecoder(PIN)
    v.calibrate()
    v.video_capture()
    GPIO.cleanup()
    print("End")
    return

if __name__ == '__main__':
    Thread(target=morse_encode).start()
    Thread(target=morse_decode).start()
