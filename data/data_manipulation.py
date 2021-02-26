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
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2





def get_vertices_list(vertices):
    '''
    This function aims to return the vertices of bbox in form of a list
    '''
    vertex_list = []
    for l in vertices:
        #print(l.x,l.y)
        vertex_list.append(l.x)
        vertex_list.append(l.y)
    return vertex_list


def get_json_files(list_of_paths):
    '''
    This function returns the json files associated to each receipt image.
    Each json file contains all the annotation infos about the receipt,
    including: descrition, languages, vertices of bbox of each word on the receipt.
    
    input:
        list_of_paths: list, all the receipt images paths have been listed 
    return:
        None: for each file in list_of_paths, the json file is saved on disk
    '''

    for j, file in tqdm(enumerate(list_of_paths)): #iterate over each file path in the list

        thejson = get_document_bounds(file) #the annotation text containing all infos of the image 
        filename = list_of_paths[j].split('/')[-1].split('.')[0].replace('-','_') # get the name '1107_receipt'

        dico = {} #empty dico
        for i in range(len(thejson)): #iterate over the length of 'thejson' 

            #create a relative dico containing two objects, in the form as example:
            #{'description':'REI', 'labels':[x: 760 y: 2374, x: 891 y: 2371, x: 892 y: 2424, x: 761 y: 2427]}
            peti_dico = {'description': thejson[i].description, 'labels': get_vertices_list(thejson[i].bounding_poly.vertices)}
            petite_list = [peti_dico] #store it into a list
            dico[f'obj{i}'] = petite_list #append to 'dico' by adding an object key and value at each iteration        
            
            #save the json file on disk
            with open(f'/home/meteor21/CUTIE_mekene/data/json_files/{filename}.json', 'w') as fp: 
                json.dump(dico, fp)

    return
        


def get_csv_files(list_of_paths):
    '''
    This function returns the csv files associated to each receipt image.
    Each csv file contains all the annotation infos about the receipt,
    including: imagename, label, vertices of bbox of each word on the receipt, imagesize.
    
    input:
        list_of_paths: list, all the receipt images paths have been listed 
    return:
        None: for each file in list_of_paths, the csv file is saved on disk
    '''

    for j, file in tqdm(enumerate(list_of_paths)): #iterate over each file path in the list

        thejson = get_document_bounds(file) #the annotation text containing all infos of the image(from google vision) 
        
        
        
        filename = list_of_paths[j].split('/')[-1].split('.')[0].replace('-','_') # get the name '1107_receipt'
        
        filename_list = [] #create an empty list
        label_list = [] #create an empty list
        image_height_list = [] #create an empty list
        image_width_list = [] #create an empty list
        X1_list = []  #create an empty list
        Y1_list = []  #create an empty list
        X2_list = []  #create an empty list
        Y2_list = []  #create an empty list
        listedebbox = [] #create an empty list
        
        
        image = PIL.Image.open(file)  #read coresponding images
        width, height = image.size  #store the dimensions
        
        
        n = len(thejson) #length of the 'thejson' return by google vision
        
        for i in range(1,n): #iteration starting from second element of 'thejson' as the first one is the whole text
            
            filename_list.append(filename) #append the filename in the preferred format
            
            label_list.append(thejson[i].description) #append the description

            image_height_list.append(image.size[1]) #append image height
            image_width_list.append(image.size[0]) #append image width
            
            #compare both x on the left side and append the smallest
            X1_list.append(min(thejson[i].bounding_poly.vertices[0].x, thejson[i].bounding_poly.vertices[3].x))
            #compare both y on the upper side and append the smallest
            Y1_list.append(min(thejson[i].bounding_poly.vertices[0].y, thejson[i].bounding_poly.vertices[1].y))
            #compare both x on the right side and append the biggest
            X2_list.append(max(thejson[i].bounding_poly.vertices[1].x, thejson[i].bounding_poly.vertices[2].x))
            #compare both y on the bottom side and append the biggest
            Y2_list.append(max(thejson[i].bounding_poly.vertices[2].y, thejson[i].bounding_poly.vertices[3].y))
            
            listedebbox.append([min(thejson[i].bounding_poly.vertices[0].x, thejson[i].bounding_poly.vertices[3].x),
                               min(thejson[i].bounding_poly.vertices[0].y, thejson[i].bounding_poly.vertices[1].y),
                               max(thejson[i].bounding_poly.vertices[1].x, thejson[i].bounding_poly.vertices[2].x),
                               max(thejson[i].bounding_poly.vertices[2].y, thejson[i].bounding_poly.vertices[3].y)])
            
            
        #create a pandas dataframe    
        df = pd.DataFrame(list(zip(filename_list, label_list, X1_list, Y1_list, X2_list, Y2_list, image_width_list, image_height_list,listedebbox)), 
                       columns =['imageName', 'labels', 'X1', 'Y1', 'X2', 'Y2', 'image_width', 'image_height', 'bbox']) 
        #save each data to a csv file on disk
        df.to_csv(f'/home/meteor21/CUTIE_mekene/data/csv_files/{filename}.csv', index = False)
        
    return 
        



def vgg_csv_receipts_onebyone(thecsv_file):
    '''
    We have used vgg annotator to manualy draw bboxes on the receipt,
    This was done in order to use IoU to keep the zones of interest. 
    This function aims to return csv files corresponding to each receipt
    They are returned under this form: 'imageName', 'X1', 'Y1', 'X2', 'Y2', 'bbox'

    Input:
        thecsv_file: the whole csv file containing the annotation done by vgg

    Output:
        save on disk, the corresponding csv to each receipt loke: 'imageName', 'X1', 'Y1', 'X2', 'Y2', 'bbox'
    '''

    vgg_200receipts = pd.read_csv(thecsv_file)  #read the csv file
    interested_frame = vgg_200receipts[['filename', 'region_shape_attributes']] #select the interested columns

    listedesnoms = [] #empty list
    listedesx1 = [] #empty list
    listedesy1 = [] #empty list
    listedesx2 = [] #empty list
    listedesy2 = [] #empty list
    listedebbox = [] #empty list

    for l in range(len(interested_frame)): #iterate over the length of the dataframe

        #append the filename
        listedesnoms.append(interested_frame.iloc[l,0].replace('.jpg','').replace('-','_'))
        #append x1
        listedesx1.append(ast.literal_eval(interested_frame.iloc[l,1])['x'])
        #append y1
        listedesy1.append(ast.literal_eval(interested_frame.iloc[l,1])['y'])
        #append x2
        listedesx2.append(ast.literal_eval(interested_frame.iloc[l,1])['x']+ast.literal_eval(interested_frame.iloc[l,1])['width'])
        #append y2
        listedesy2.append(ast.literal_eval(interested_frame.iloc[l,1])['y']+ast.literal_eval(interested_frame.iloc[l,1])['height'])
        #append the bbox in form of [x1,y1,x2,y2]
        listedebbox.append([ast.literal_eval(interested_frame.iloc[l,1])['x'],
                            ast.literal_eval(interested_frame.iloc[l,1])['y'],
                            ast.literal_eval(interested_frame.iloc[l,1])['x']+ast.literal_eval(interested_frame.iloc[l,1])['width'],
                            ast.literal_eval(interested_frame.iloc[l,1])['y']+ast.literal_eval(interested_frame.iloc[l,1])['height']])
        

    #save in to a big dataframe
    deaf = pd.DataFrame(list(zip(listedesnoms, listedesx1, listedesy1, listedesx2, listedesy2,listedebbox)), 
                           columns =['imageName', 'X1', 'Y1', 'X2', 'Y2', 'bbox']) 

    for t in tqdm(range(len(deaf))):  #iterate over the length of the dataframe

        #save each data to a csv file on disk
        deaf.loc[[t]].to_csv(f'/home/meteor21/CUTIE_mekene/data/200receipts_csv_onebyone/{deaf.iloc[t,0]}.csv', index = False)
   
    return




def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Input:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon) #to avoid dividing by 0
    return iou



def get_masks(start,end):
    '''
    This function aims to save on disk, the corresponding masks with the zone of interest of each receipt image.
    The 200 images are saved on disk according to an ascending numerical order from 1000 to 1199.
    To make the computation easy, we've done the iteration over each number in this range.
    Input:
        start: the number of the first image (example:start = 1000)
        end: the number after the one of the last image (example:end = 1200 )
    return:
        save each corresponding masks on disk
    '''

    for z in tqdm(range(start,end)): #iterate over this range
        
        #read the corresponding single row csv file generated by vgg manual annotation 
        small_csv = pd.read_csv(f'/home/meteor21/CUTIE_mekene/data/200receipts_csv_onebyone/{z}_receipt.csv')
        #read the corresponding large csv file generated by the google vision annotation
        big_csv = pd.read_csv(f'/home/meteor21/CUTIE_mekene/data/csv_files/{z}_receipt.csv')
        #make zero matrix with size corresponding to the image size
        matrix = (np.zeros(((big_csv.iloc[0, 7]), (big_csv.iloc[0, 6]))))


        for i in range(len(big_csv)): #iterate over the length of the larger csv file 
            
            #make zero matrix with size corresponding to the image size
            block = (np.zeros(((big_csv.iloc[i, 7]), (big_csv.iloc[i, 6]))))
            #compute the IoU of the bbox in the small csv with each bbox in the big csv
            theIoU = (get_iou(ast.literal_eval(small_csv.iloc[0,-1]), ast.literal_eval(big_csv.iloc[i,-1])))
            
            if theIoU > 1e-5: #In case the IoU is larger than epsilon(1e-5)
                #fill with 1s the corresponding rectangle in the 'block' array 
                block[(big_csv.iloc[i, 3]):(big_csv.iloc[i, 5]), (big_csv.iloc[i, 2]):(big_csv.iloc[i, 4])] = 1

            matrix = np.add(matrix, block) #matrix + block
       
        #avoid overlap of small rectangles on 'matrix' and also keep the values of 'matrix' to (0.0 and 1.0)
        matrix[matrix>1] = 0


        #save the image in jpg
        img = Image.fromarray(matrix)
        img = img.convert('RGB')
        img.save(f'/home/meteor21/CUTIE_mekene/data/200receipts_masks/{z}-receipt.jpg')

    return




def vizz_of_overlay_rectangles(start, end):
    '''
    This function aims to visualize each receipt with the corresponding zone of interest, 
    clearly overlayed by a transparent red rectangle. 
    The 200 images are saved on disk according to an ascending numerical order from 1000 to 1199.
    To make the computation easy, we've done the iteration over each number in this range.
    Input:
        start: the number of the first image (example:start = 1000)
        end: the number after the one of the last image (example:end = 1200 )
    return:
        visualisation of the images 
    '''
    

    for z in range(start,end): #iterate over this range

        #read the corresponding single row csv file generated by vgg manual annotation 
        small_csv = pd.read_csv(f'/home/meteor21/CUTIE_mekene/data/200receipts_csv_onebyone/{z}_receipt.csv')
        #read the corresponding large csv file generated by the google vision annotation
        big_csv = pd.read_csv(f'/home/meteor21/CUTIE_mekene/data/csv_files/{z}_receipt.csv')
        #read the corresponding image
        image = Image.open(f'/home/meteor21/CUTIE_mekene/large-receipt-image-dataset-SRD/{z}-receipt.jpg')


        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)

        for i in range(len(big_csv)): #iterate over the length of the large csv file 
            
            #compute the IoU of the bbox in the small csv with each bbox in the big csv
            theIoU = (get_iou(ast.literal_eval(small_csv.iloc[0,-1]), ast.literal_eval(big_csv.iloc[i,-1])))
            
            if theIoU > 1e-5: #In case the IoU is larger than epsilon(1e-5)

                # Create a Rectangle patch
                rect = patches.Rectangle(((big_csv.iloc[i, 2]),(big_csv.iloc[i, 3])),
                                         ((big_csv.iloc[i, 4])-(big_csv.iloc[i, 2])),
                                         ((big_csv.iloc[i, 5])-(big_csv.iloc[i, 3])),
                                         linewidth=1, edgecolor='r', facecolor='r', alpha=0.2)

                # Add the patch to the Axes
                ax.add_patch(rect)

        plt.show()  #visualise
        
    return




def resize_image(image_path, desired_dimension):
    '''
    This function aims to resize the images into square format without affecting the previous annotations
    Input:
        image_path:the image path on disk
        desired_dimension:if the target shape we want is (n,n), desired_dimension = n
    Return: the reshaped image in np.array format

    '''
    

    al = plt.imread(image_path) #read the image


    if len(al.shape) == 2: #check if it is a 1 channels image
        

        al = cv2.merge((al,al,al)) #merge all channels together


    elif al.shape[2] == 4: b #check if it is a 4 channels image

        al = cv2.cvtColor(al, cv2.COLOR_BGRA2BGR) #convert it into 3 channels image

    else:  #otherwise

        al = al


    w = al.shape[1]  #store the width
    h = al.shape[0]  #store the height
    c = al.shape[2]  #store the number of channels(3 in this case)

    #print(w)
    #print(h)
    #print(c)

    # In case image is horizontally orientated
    if w > h:
        combine = np.zeros((w,w,c))
        combino = combine
        for i in range(c):
            combino[0:h ,: ,i] = al[:,:,i]
        resized = cv2.resize(combino, (desired_dimension, desired_dimension), interpolation=cv2.INTER_NEAREST)

    # In case image is vertically orientated
    elif w < h:
        combine = np.zeros((h,h,c))
        combino = combine
        for i in range(c):
            combino[: ,0:w,i] = al[:,:,i]
        resized = cv2.resize(combino, (desired_dimension, desired_dimension), interpolation=cv2.INTER_NEAREST)

    # In case image is squared
    else:
        resized = cv2.resize(al, (desired_dimension, desired_dimension), interpolation=cv2.INTER_NEAREST)

    al = resized 

    plt.imshow(al.astype(np.uint8))  # to Clip input data to the valid range for imshow with RGB data.
    plt.show()
    #print('the final shape is:',al.shape)

    return al





def visualize_bbox_centers(csv_file, desired_shape):
    '''
    This function aims to visualise the center of each bbox on the corresponding image
    input:
        csv_file: the path the csv file
        desired_shape: if the target shape we want is (n,n), desired_dimension = n
    return:
        show the plot of the numpy array with each bbox and its center 
    '''
    
    receipt = pd.read_csv(csv_file)
    maximum = max(receipt.iloc[1,6],receipt.iloc[1,7]) #maximum between height or width of the corresponding image

    alal = np.zeros((maximum,maximum)) #0s numpy array

    for i in range(len(receipt)): #iterate over the length of the csv file
        alal[receipt.iloc[i,3]:receipt.iloc[i,5],receipt.iloc[i,2]:receipt.iloc[i,4]] = 1 #fill th bbox to 1s


    Xc_list = [] #empty list
    Yc_list = [] #empty list
    ratio = desired_shape/maximum #target over actual

    for i in range(len(receipt)): #iterate over the length of the csv file
        Xc = int(ratio*(receipt.iloc[i,2]+receipt.iloc[i,4])/2) #the x coordinate of the center
        Yc = int(ratio*(receipt.iloc[i,3]+receipt.iloc[i,5])/2) #the y coordinate of the center  
        Xc_list.append(Xc) #append to a list
        Yc_list.append(Yc) #append to a list

    receipt['Xc'] = Xc_list #add this column to the dataframe
    receipt['Yc'] = Yc_list #add this column to the dataframe

    #draw the center of each bbox
    for j in range(len(receipt)):

        plt.scatter([receipt.iloc[j,9]], [receipt.iloc[j,10]],c='r',s=5)

    alal = cv2.resize(alal, (desired_shape, desired_shape), interpolation=cv2.INTER_NEAREST)
    implot = plt.imshow(alal)

    plt.show()
    
    return





def get_relative_centers(csv_file, desired_shape):
    '''
    This function aims to give back relative position of the center of each bbox on the corresponding image
    input:
        csv_file: the path the csv file
        desired_shape: if the target shape we want for the image is (n,n), desired_dimension = n
    return:
        show the plot of the numpy array with each bbox and its center 
    '''
    
    receipt = pd.read_csv(csv_file)
    maximum = max(receipt.iloc[1,6],receipt.iloc[1,7]) #maximum between height or width of the corresponding image

    width_and_height = [] #empty list
    Xc_list = [] #empty list
    Yc_list = [] #empty list
    ratio = desired_shape/maximum #target over actual

    for i in range(len(receipt)): #iterate over the length of the csv file
        Xc = int(ratio*(receipt.iloc[i,2]+receipt.iloc[i,4])/2) #the x coordinate of the center
        Yc = int(ratio*(receipt.iloc[i,3]+receipt.iloc[i,5])/2) #the y coordinate of the center  
        Xc_list.append(Xc) #append to a list
        Yc_list.append(Yc) #append to a list
        width_and_height.append(maximum) #append to a list

    
    receipt['Xc'] = Xc_list #add this column to the dataframe
    receipt['Yc'] = Yc_list #add this column to the dataframe
    receipt['New_Image_width_and_height'] = width_and_height #add this column to the dataframe
   

    #save each data to a csv file on disk
    receipt.to_csv(f'/home/meteor21/CUTIE_mekene/data/csv_files_with_centers/{receipt.iloc[1,0]}_centers.csv', index = False)
   
    return




def initial_grids(csv_file, grid_size, desired_image_size):
    '''
    This function aims to return the grid (numpy array) corresponding to each receipt image.
    It saves the bbox centers to 1s and the locations where there's no text to 0s
    By taking in consideration the extrem case where overlap can occur, 
    Due to the reduced size of the grid compared to the corresponding desired image receipt size.
    
    Input:
        csv_file: the path of the csv file containing infos about the bbox centers 
        grid_size: if the target shape we want for the grid is (n,n), grid_size = n
        desired_image_size: if the target shape we want for the image is (P,P), desired_image_size = P
        
    Return:
        show the grind(numpy array) with every bbox center not overlapping
    '''
    
    csv_center = pd.read_csv(csv_file) #read csv file
    ratio = desired_image_size/grid_size #compute the ratio 
    orh = np.zeros((int(grid_size),int(grid_size))) #0s numpy array


    for j in range(len(csv_center)): #iterate over the length of the csv file 
        
        #Check if the actual loaction of the center relative to the receipt is empty
        if orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1] ==0:
            #if yes, fill it to 1
            orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]+=1

        #if no, check the right side 
        elif orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]==1:
            if orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio)+1:int(csv_center.iloc[j,9]/ratio)+2]==0:
                orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio)+1:int(csv_center.iloc[j,9]/ratio)+2]+=1
            
            #if no, check the down side 
            elif orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio)+1:int(csv_center.iloc[j,9]/ratio)+2]==1:
                if orh[int(csv_center.iloc[j,10]/ratio)+1:int(csv_center.iloc[j,10]/ratio)+2,int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]==0:     
                    orh[int(csv_center.iloc[j,10]/ratio)+1:int(csv_center.iloc[j,10]/ratio)+2,int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]+=1
                
                #if no, check the left side 
                elif orh[int(csv_center.iloc[j,10]/ratio)+1:int(csv_center.iloc[j,10]/ratio)+2,int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]==1:
                    if orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio)-1:int(csv_center.iloc[j,9]/ratio)]==0:
                        orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio)-1:int(csv_center.iloc[j,9]/ratio)]+=1
                    #if no, check the upper side 
                    elif orh[int(csv_center.iloc[j,10]/ratio):int(csv_center.iloc[j,10]/ratio)+1,int(csv_center.iloc[j,9]/ratio)-1:int(csv_center.iloc[j,9]/ratio)]==1:
                        if orh[int(csv_center.iloc[j,10]/ratio)-1:int(csv_center.iloc[j,10]/ratio),int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]==0:
                            orh[int(csv_center.iloc[j,10]/ratio)-1:int(csv_center.iloc[j,10]/ratio),int(csv_center.iloc[j,9]/ratio):int(csv_center.iloc[j,9]/ratio)+1]+=1


     
    

    if max(np.unique(orh)) != 1. :
        print(np.unique(orh))
        print(csv_file)
        
    plt.imshow(orh)
    plt.show()
    
    return