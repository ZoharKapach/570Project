#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:34:53 2019

@author: zkapach
"""

import csv
from collections import Counter, OrderedDict
from PIL import Image
import cv2

num_classes = 16
#
#with open("file_counts.csv") as file_counts, open("classes.csv") as classes, open("locations.csv") as locations:
#    fc_reader = csv.reader(file_counts, delimiter=',')
#    cls_reader = csv.reader(classes, delimiter=',')
#    locs_reader = csv.reader(locations, delimiter=',')
#    print(fc_reader)
#    line_count = 0
#    with open("gt_recounts.csv", "w") as gt_file:
#        gt_writer = csv.writer(gt_file)
#        
#        for row1, row2, row3 in zip(fc_reader, locs_reader, cls_reader):
#            if "classes" not in row3:
#                clean_class_row = [float(i) for i in row3 if i]
#                clean_coord_row = [float(i) for i in row2 if i]
#                if(len(clean_coord_row) == 1):
#                    clean_coord_row = [-1]
#                
#                if (len(clean_coord_row) > 1):
#                    new_coord_row = []
#                    y_check = False
#                    for idx, item in enumerate(clean_coord_row):
#                        if y_check:
#                            y_check = False
#                            continue
#                        new_coord_row.append((clean_coord_row[idx+ 1], clean_coord_row[idx]))
#                        y_check = True 
#                    img = cv2.imread('/home/devs/570/zproject/locating-objects-classes/mpii_human_pose_v1/images/'+file_name)
#                    old_x, old_y = img.shape[0], img.shape[1]
#                    for idx, coord in enumerate(new_coord_row):
#                        y = round(int(round(coord[0]))*256/old_x)
#                        x = round(int(round(coord[1]))*256/old_y)
#                        coord = (y, x)
#                        new_coord_row[idx] = coord
#                        
#                else:
#                    new_coord_row = clean_coord_row
#                
#                if row1[1] != '0':
#                    classes_dict = OrderedDict(Counter(clean_class_row))
#    #                print(list(classes_dict.values()))
#                    
#                    for idx in range(num_classes):
#                        if idx not in classes_dict.keys():
#                            classes_dict[idx] = 0
#                            
#                    row1[1] = list(classes_dict.values())
#                    
#                new_final_row = row1 + [new_coord_row] + [clean_class_row]
#                
#            else:
#                row2 = [i for i in row2 if i]
#                row3 = [i for i in row3 if i]
#                new_final_row = row1+row2+row3
##            file_name = new_final_row[0]
##            
##            if file_name != 'filename' and new_final_row[1] != '0' :
##                
##                old_x, old_y = img.shape[0], img.shape[1]
##                img = cv2.resize(img, (256, 256))
##                color = [255, 0, 0]
##                print("here")
##                for coord in new_coord_row:
##                    y = round(int(round(coord[0]))*256/old_x)
##                    x = round(int(round(coord[1]))*256/old_y)
##                    img = cv2.circle(img, (x, y), 3, color, -1)
##                
##                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
##                cv2.resizeWindow('image', 600,600)
##                cv2.imshow("image", img)
##                cv2.waitKey()
##                break
#            
#            if new_final_row[1] != '0':    
#                gt_writer.writerow(new_final_row)
#
#gt_file.close()
#file_counts.close()
#classes.close()
#locations.close()
#    













#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:34:53 2019

@author: zkapach
"""

import csv

with open("file_counts.csv") as file_counts, open("classes.csv") as classes, open("locations.csv") as locations:
    fc_reader = csv.reader(file_counts, delimiter=',')
    cls_reader = csv.reader(classes, delimiter=',')
    locs_reader = csv.reader(locations, delimiter=',')
    print(fc_reader)
    line_count = 0
    with open("final_test1_gt.csv", "w") as gt_file:
        gt_writer = csv.writer(gt_file)
        
        for row1, row2, row3 in zip(fc_reader, locs_reader, cls_reader):
            if row1[0] == '033324174.jpg':
                if "classes" not in row3:
                    clean_class_row = [float(i) for i in row3 if i]
                    clean_coord_row = [float(i) for i in row2 if i]
                    if(len(clean_coord_row) == 1):
                        clean_coord_row = [-1]
                    
                    if (len(clean_coord_row) > 1):
                        new_coord_row = []
                        y_check = False
                        for idx, item in enumerate(clean_coord_row):
                            if y_check:
                                y_check = False
                                continue
                            new_coord_row.append((clean_coord_row[idx+ 1], clean_coord_row[idx]))
                            y_check = True 
                        file_name = row1[0]
                        img = cv2.imread('/home/devs/570/zproject/locating-objects-classes/mpii_human_pose_v1/images/'+file_name)
                        old_x, old_y = img.shape[0], img.shape[1]
                        img = cv2.resize(img, (256, 256))
                        for idx, coord in enumerate(new_coord_row):
                            y = round(int(round(coord[0]))*256/old_x)
                            x = round(int(round(coord[1]))*256/old_y)
                            coord = (y, x)
                            new_coord_row[idx] = coord
                            img = cv2.circle(img, (x, y), 3, [0,0,255], -1)
                        
                        cv2.imwrite('mpii_image_example.png',img)
                    else:
                        new_coord_row = clean_coord_row
                    
                    new_final_row = row1 + [new_coord_row] + [clean_class_row]
                    
                else:
                    row2 = [i for i in row2 if i]
                    row3 = [i for i in row3 if i]
                    new_final_row = row1+row2+row3
                if new_final_row[1] != '0':    
                    gt_writer.writerow(new_final_row)

gt_file.close()
file_counts.close()
classes.close()
locations.close()
    
