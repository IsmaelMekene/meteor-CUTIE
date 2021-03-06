'''
@Author: Mékéné
@Github: https://github.com/IsmaelMekene
@Project: https://github.com/luyanger1799/meteor-CUTIE
'''




import numpy as np
import pandas as pd 
import glob2
import pickle
import json
import argparse
from enum import Enum
import io
from tqdm import tqdm 

from google.cloud import vision
from PIL import Image, ImageDraw




def get_document_bounds(image_file):#, feature):
    """
    This function returns document bounds given an image.
    
    input:
        image_file: the image file path
    return:
        text: the annotation text containing all infos of the image  
        
    
    Warning!!!
    Set the environment variable by storing in it,
    the JSON file that contains your service account key.
    
    In our case:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/meteor21/google-cloud-sdk/bin/meteorCUTIE-d63f1b40262d.json'
    """
    
    bounds = [] #empty list

    with io.open(image_file, 'rb') as image_file: #read the image
        content = image_file.read() 

    image = vision.Image(content=content) #pass it to the google 'vision' function 

    
    
    client = vision.ImageAnnotatorClient() #GOOGLE_APPLICATION_CREDENTIALS
    response = client.text_detection(image=image) #loggings
    text = response.text_annotations #the text containing the annotation infos
    del response     # to clean-up the system memory

    return text
