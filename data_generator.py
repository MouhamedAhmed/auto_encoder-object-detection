import numpy as np
import random
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv

img_size = 64
# create the directory of images
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'images')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

for i in range(5000):
    # randomize background color
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    back_color = [r,g,b]

    img = np.ones((img_size,img_size,3)).astype(int)*back_color

    # forground
    # randomize foreground color
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    fore_color = [r,g,b]
    # randomize center
    center_x = random.randint(img_size//8,(7 * img_size)//8)
    center_y = random.randint(img_size//8,(7 * img_size)//8)
    radius = random.randint(img_size//10,np.min([img_size - 1 - center_x, img_size - 1 - center_y, center_x, center_y]))

    rr,cc = skimage.draw.circle(center_x, center_y, radius)
    img[rr,cc] = fore_color

    plt.imsave('images/' + str(i)+'.jpg', img.astype("uint8"))

    # csv save
    img_directory = os.path.join(final_directory, str(i)+'.jpg')
    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius
    y2 = center_y + radius

    with open('bb.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_directory, x1, y1, x2, y2])


