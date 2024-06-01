from tkinter import *
from PIL import Image, ImageDraw
from tkinter import messagebox
import tensorflow.keras
import cv2
import numpy as np

# Путь к сохраненной модели
model_path = r'model_40ep.keras'
# Загрузка сохраненной модели
model = tensorflow.keras.models.load_model(model_path)


def draw(event):
    x1, y1 = (event.x - bruch_size), (event.y - bruch_size)
    x2, y2 = (event.x + bruch_size), (event.y + bruch_size)
    canvas.create_oval(x1, y1, x2, y2, fill=color, width=0)
    draw_img.ellipse((x1, y1, x2, y2), fill=color, width=0)


def clear_canvas():
    canvas.delete('all')
    canvas['bg'] = 'white'
    # global draw_img
    draw_img.rectangle((0, 0, 512, 512), width=0, fill='white')


def save_img():
    global image1
    filename = f'image.png'
    resized_image = image1.resize((256, 256))
    resized_image.save(filename)


def predict_letter():
    save_img()
    image_path = f'image.png'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, (1, 256, 256, 1))
    img = img / 255.0
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    cyrillic_lower_letters = 'ёабвгдежзийклмнопрстуфхцчшщъыьэюя'
    messagebox.showinfo('Распознавание', f'буква: {cyrillic_lower_letters[predicted_class]}')
    #print(predicted_class)


x = 0
y = 0

root = Tk()
root.title("Paint")
root.geometry("512x512")
root.resizable(True, True)

bruch_size = 13
color = 'black'

root.columnconfigure(6, weight=1)
root.rowconfigure(2, weight=1)
canvas = Canvas(root, bg='white')
canvas.grid(row=2, column=0, columnspan=7, padx=5, pady=5, sticky=E + W + S + N)
canvas.bind("<B1-Motion>", draw)
image1 = Image.new('RGB', (512, 512), 'white')
draw_img = ImageDraw.Draw(image1)

predict_button = Button(root, text='Распознать', width=15, command=predict_letter)
predict_button.grid(row=1, column=3, padx=6)

Label(root, text='Действия:').grid(row=1, column=0, padx=6)

Button(root, text='Очистить', width=10, command=clear_canvas).grid(row=1, column=2, padx=6)

#Button(root, text='Сохранить', width=10, command=save_img).grid(row=1, column=5, padx=6)
root.mainloop()
