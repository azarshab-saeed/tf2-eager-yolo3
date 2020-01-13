import xml.etree.cElementTree as ET
import random
from random import randrange
import numpy as np


def create_numbers(save_path,name_of_file,numbers, number_labels,nois_data, maxlength_x=1, maxlength_y=1, digit_sz=(28, 28)):
    # Randomly choose a number length:
    img_len = np.random.choice(range(maxlength_x*maxlength_y)) + 1
    label = np.empty(maxlength_x*maxlength_y, dtype='str')

    # Randomly choose where in our image the sequence of numbers will appear
    if img_len < maxlength_x:
        st_point = np.random.choice(maxlength_x - img_len)
    else:
        st_point = 0
    
    charmap = np.zeros(maxlength_x*maxlength_y)
    random_index =np.random.randint(0, len(charmap), size=(img_len))
    charmap[random_index] = 1

    # Define a blank character - this will ensure our input image always have the same dimensions
    #blank_char = np.zeros_like(digit_sz)
    blank_lbl = "."

    # Initialize a blank image with maxlen * digit_dz width and digit_sz height
    new_img_len = maxlength_x * digit_sz[1]
    new_img = np.zeros((digit_sz[0]*maxlength_y, new_img_len))
    #creat annotation xml file
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text =str(name_of_file)+".png"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(maxlength_x * digit_sz[1])
    ET.SubElement(size, "height").text = str(digit_sz[0]*maxlength_y)
    ET.SubElement(size, "depth").text = "1"
    
    # Fill in the image with random numbers from dataset, starting at st_point
    j=0
    k=0
    for i, b in enumerate(charmap):
        
        if(i>(maxlength_x*(j+1)-1)):
            j=j+1
            k=0
        if b > 0:
            n = np.random.choice(len(numbers))
            st_pos_x = k * digit_sz[1]
            st_pos_y = j * digit_sz[1]
            new_img[st_pos_y:st_pos_y + digit_sz[1], st_pos_x:st_pos_x + digit_sz[1]] = cv2.subtract(255, numbers[n])
            img=cv2.subtract(255, numbers[n])
            ret,thresh = cv2.threshold(img,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            #print ("number of countours detected before filtering %d -> "%len(contours))
            for item in range(len(contours)):
                cnt = contours[item]
                if len(cnt)>5:
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    #plt.figure(figsize=(1,1))
                    #print(x,y,w,h)
                    
            #imgplot = plt.imshow(cv2.subtract(255, numbers[n]))
            #plt.show()
            #print(x,y,w,h)
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(number_labels[n])
            bndbox= ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(st_pos_x+x)
            ET.SubElement(bndbox, "ymin").text = str(st_pos_y+y)
            ET.SubElement(bndbox, "xmax").text = str((st_pos_x+x+w))
            ET.SubElement(bndbox, "ymax").text = str((st_pos_y+y+h))
            
        else:
            n = np.random.choice(len(nois_data))
            st_pos_x = k * digit_sz[1]
            st_pos_y = j * digit_sz[1]
            new_img[st_pos_y:st_pos_y + digit_sz[1], st_pos_x:st_pos_x + digit_sz[1]] = nois_data[n]
            img=cv2.subtract(255, numbers[n])
            ret,thresh = cv2.threshold(img,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            #print ("number of countours detected before filtering %d -> "%len(contours))
            for item in range(len(contours)):
                cnt = contours[item]
                if len(cnt)>5:
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    #plt.figure(figsize=(1,1))
                    #print(x,y,w,h)
                    
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = "noise"
            bndbox= ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(st_pos_x+x)
            ET.SubElement(bndbox, "ymin").text = str(st_pos_y+y)
            ET.SubElement(bndbox, "xmax").text = str((st_pos_x+x+w))
            ET.SubElement(bndbox, "ymax").text = str((st_pos_y+y+h))
            
        k=k+1
    tree = ET.ElementTree(root)
    file=save_path+name_of_file
    tree.write(file+".xml")
    return new_img
    
import numpy as np
from keras.datasets.mnist import load_data
import glob
from PIL import Image
import cv2
import random
from random import randrange
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
#data = np.load('mnist.npz')
#testX,trainX, trainy, testy =data['x_test'], data['x_train'], data['y_train'], data['y_test']
#load noise data
path = 'noise_data/'
image_list = []
for filename in glob.glob(path + '/*.*'): #assuming gif
    im=cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image_list.append(im)
###############################
save_path = "dataset/mnist/anns/"
save_path_1 = "dataset/mnist/imgs/"
number_of_generat_data=100
for i in range(number_of_generat_data):
    file_name="image"+str(i)
    img = create_numbers(save_path,file_name ,trainX, trainy,image_list)
    cv2.imwrite(save_path_1+file_name+'.png', img)
