from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

print("Starting in 4 seconds...")
time.sleep(4)

while True:
    keyboard.press(Key.down)
    keyboard.release(Key.down)

    keyboard.press(Key.space)
    keyboard.release(Key.space)

    time.sleep(0.00001)
