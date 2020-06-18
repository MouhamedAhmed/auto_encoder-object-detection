import numpy as np
import random
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv


# create the directory of images
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'images')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

for i in range(10000):
    # randomize background color
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    back_color = [r,g,b]

    img = np.ones((256,256,3)).astype(int)*back_color

    # forground
    # randomize foreground color
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    fore_color = [r,g,b]
    # randomize center
    center_x = random.randint(20,235)
    center_y = random.randint(20,235)
    radius = random.randint(20,np.min([255 - center_x, 255 - center_y, center_x, center_y]))

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


