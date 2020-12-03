import cv2
import numpy as np
import math
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import pytesseract


import logging
log_format = '%(asctime)3s %(lineno)d  %(levelname)5s  %(message)s'

logging.basicConfig(level=logging.DEBUG,
                    format=log_format,
                    filename='app.log',
                    filemode='w')
import re

def extract_main_table(gray_image):
#    inverted = cv2.bitwise_not(gray_image)
#    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
#
#    thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
##    image, cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#    cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Simple threshold
    _, thr = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)    
    # Morphological closing to improve mask
    close = cv2.morphologyEx(255 - thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))    
    # Find only outer contours
    cnts= cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#    cnts = cnts[1]# if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    areaThr = 150000
    area = cv2.contourArea(cnts[0])
    if area > areaThr:
        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    #    pad_box = np.array([[i[0]+10,i[1]+10] for i in box])
        box[0] = box[0]-5
        box[1] = box[1]-5
        box[2] = box[2]+5
        box[3] = box[3]+5  
        
        extracted = four_point_transform(gray_image.copy(), box.reshape(4, 2))
    #    plt.imshow(extracted)
        return extracted
    else: 
        return None

def horizontal_boxes_filter(box,width):
    x,y,w,h = box
    return w > width * 0.2

def vertical_boxes_filter(box,height):
    x,y,w,h = box
    return  h > height * 0.2


def extract_rows_columns(gray_image):
    inverted = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    height, width = gray_image.shape

    thresholded = cv2.threshold(blurred, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    vertical_kernel_height = math.ceil(height*0.3)
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontal_kernel_width = math.ceil(width*0.3)
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(thresholded, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)
    _,vertical_contours, _ = cv2.findContours(verticle_lines_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (vertical_contours, vertical_bounding_boxes) = sort_contours(vertical_contours, method="left-to-right")

    filtered_vertical_bounding_boxes = list(filter(lambda x:vertical_boxes_filter(x,height), vertical_bounding_boxes))

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(thresholded, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)
    _,horizontal_contours, _ = cv2.findContours(horizontal_lines_img.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    horizontal_contours, horizontal_bounding_boxes = sort_contours(horizontal_contours, method="top-to-bottom")
    filtered_horizontal_bounding_boxes = list(filter(lambda x:horizontal_boxes_filter(x,width), horizontal_bounding_boxes))
    extracted_rows_columns = []
    for idx_h, horizontal_bounding_box in enumerate(filtered_horizontal_bounding_boxes):
        if idx_h == 0:
            continue
        hx_p,hy_p,hw_p,hh_p = filtered_horizontal_bounding_boxes[idx_h-1] #previous horizontal box
        hx_c,hy_c,hw_c,hh_c = horizontal_bounding_box
        extracted_columns = []
        for idx_v, vertical_bounding_box in enumerate(filtered_vertical_bounding_boxes):
            if idx_v == 0:
                continue
            vx_p,vy_p,vw_p,vh_p = filtered_vertical_bounding_boxes[idx_v-1] #previous horizontal box
            vx_c,vy_c,vw_c,vh_c = vertical_bounding_box
            table_cell = gray_image[hy_p:hy_c+hh_c,vx_p:vx_c+vw_c]
            blurred = cv2.GaussianBlur(table_cell, (5, 5), 0)
            thresholded = cv2.threshold(blurred, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            _,contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            extracted = four_point_transform(table_cell.copy(), box.reshape(4, 2))[1:-1,1:-1] #remove 1 px from each side
            ret,extracted = cv2.threshold(extracted,165,255,cv2.THRESH_BINARY)
            extracted_columns.append(extracted)
        extracted_rows_columns.append(extracted_columns)
    return extracted_rows_columns

def is_table(image):
    try:
        extracted_table = extract_main_table(image)
        if extracted_table.any() :
            logging.info('Table found')
#            print("table Detected")
            return True
        else:
#            print("table 1Detected")
            return False
    except Exception as e:
#        print(e)
        logging.info('Error in Table Detection: %s',e)
#        print("Error")
        return False
        pass
    
def Get_table_Extraction(gray_image):
#    config_table = ("-l eng+ara --oem 1 --psm 6")
    config_table = ("-l eng --oem 1 --psm 6")
    extracted_table = extract_main_table(gray_image)
    row_images = extract_rows_columns(extracted_table) #[1:]
    idx = 0
    all_rows = []
    for row in row_images:
        idx += 1
#        print("%s : Extracting row %d out of %d " % (file_name, idx,len(row_images)))
        row_texts = []
        for column in row:
            try:
                text = pytesseract.image_to_string(column, config=config_table)
                text = re.sub(r'[^a-zA-Z0-9,.-]', ' ',text)
                row_texts.append(text)
            except Exception as e:
                logging.info('Error in Table data extraction : %s', e)
#                print(e)
                pass    
#        print(row_texts)
        all_rows.append(row_texts)
        logging.info('Table Data rowwise: %s',all_rows)
    return all_rows
