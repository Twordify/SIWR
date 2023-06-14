import os
from pathlib import Path
import argparse
import re
import numpy as np
import cv2
import math
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from matplotlib import pyplot as plt

photos = []

class BBox:
    def __init__(self, points, box_crop_points, height, width, diagonal, histograms, name, box_count ):
        self.points = list(points)  #list of points defining bboxes
        self.box_crop_points = list(box_crop_points) #list of points of cropped bboxes
        self.height = list(height)      #list of bbox height
        self.width = list(width)    #list of bbox width
        self.diagonal = list(diagonal)    #list of bbox diagonal
        self.histograms = list(histograms)  #list with histograms
        self.name = name #name of picture
        self.box_count = box_count #how many bboxes on picture

#source: https://www.geeksforgeeks.org/combinations-in-python-without-using-itertools/

def preparation(search, r):
    name = search
    length = len(name)
    if r> length:
        return
    id = np.arange(r)
    yield tuple(name[i] for i in id)
    while True:

        for i in reversed(range(r)):
            if id[i] != i + length - r:
                break
        else:
            return

        id[i] +=1
        for ii in range(i+1, r):
            id[ii] = id[ii-1]+1
        yield tuple(name[i] for i in id)


def read(directory):
    plik = str(directory) + '/bboxes.txt'
    with open(plik) as f:
        lines = f.readlines()
    subs = None

    for line in lines:
        subs = line[:1]
        break

    #variables added to be part of graph
    name_img_cur = '^' + str(subs)
    current_photo_flag = True
    count = 0
    points = []
    box = []
    height = []
    width = []
    diagonal = []
    histograms = []

    photo_name = None
    photo_bbox_count = None



    for line in lines:
        if current_photo_flag:
            result = re.match(name_img_cur, line)
            if result:

                #clear variables before adding new photo
                current_photo_flag = False
                pp = line[:-1]
                photo_name = str(directory) + '/frames/' + str(pp)
                img = cv2.imread(photo_name)
                width.clear()
                diagonal.clear()
                histograms.clear()
                points.clear()
                box.clear()
                height.clear()

        else:

            if len(line) < 3 and line != '\n':
                count = int(line) #how many next lines should be read
                photo_bbox_count = count

            else:
                if count !=0:
                    data = line.split()
                    points.append(float(data[0]))
                    points.append(float(data[1]))
                    points.append(float(data[2]))
                    points.append(float(data[3]))
                    x = float(data[0])  #x1
                    y = float(data[1])  #y1
                    w = float(data[2])  #x2 = (x+w)
                    h = float(data[3])  #y2 = (y+h)

                    count-=1

                    #cut bbox with certain person
                    crop = img[int(y):int(y + h), int(x):int(x + w)]

                    #append box to class
                    box.append(crop)

                    #get lenght, width and diagonal
                    w_diagonal = math.sqrt(pow((x - x - w), 2) + pow((y - h - y),2)) 
                    w_width = math.sqrt(pow((x - x - w), 2))
                    w_height = math.sqrt(pow((y - h - y), 2))

                    #append values
                    diagonal.append(w_diagonal)
                    width.append(w_width)
                    height.append(w_height)

                    #Get one-third of these values
                    w_third_hist_width = w_width / 3
                    w_third_hist_hight = w_height / 3

                    #Naive way to crop bacground 
                    hist_crop = img[int(y + w_third_hist_hight):int(y + h - w_third_hist_hight), int(x+ w_third_hist_width):int(x + w - w_third_hist_width)]

                    #histogram of cropped image
                    histg = cv2.calcHist([hist_crop], [0], None, [256], [0, 256])
                    histograms.append(histg)

                    if count == 0:
                        current_photo_flag = True
                        #append parameters to class
                        to_class = BBox(points, box, height, width, diagonal, histograms, photo_name, photo_bbox_count)
                        photos.append(to_class)



#compare histograms of cropped images
def compare_hist(img_1, bbox_img_1, img_2, bbox_img_2):
    hist_1 = photos[img_1].histograms[bbox_img_1]
    hist_2 = photos[img_2].histograms[bbox_img_2]

    bhattacharyya_distance = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
    matching_result = cv2.matchTemplate(hist_1, hist_2, cv2.TM_CCOEFF_NORMED)[0][0]
    similarity_score = 1 - matching_result
    final_score = (bhattacharyya_distance / 3) + similarity_score

    return 1 - final_score



def compare_dims(img_1, bbox_img_1, img_2, bbox_img_2):
    H_bb_1 = photos[img_1].height[bbox_img_1]
    H_bb_2 = photos[img_2].height[bbox_img_2]

    W_bb_1 = photos[img_1].height[bbox_img_1]
    W_bb_2 = photos[img_2].height[bbox_img_2]

    D_bb_1 = photos[img_1].diagonal[bbox_img_1]
    D_bb_2 = photos[img_2].diagonal[bbox_img_2]

    #get dimensions ratio to have values lower than 1
    
    ratio_H = H_bb_2 / H_bb_1 if H_bb_1 > H_bb_2 else H_bb_1 / H_bb_2
    ratio_W = W_bb_2 / W_bb_1 if W_bb_1 > W_bb_2 else W_bb_1 / W_bb_2
    ratio_D = D_bb_2 / D_bb_1 if D_bb_1 > D_bb_2 else D_bb_1 / D_bb_2

    return ratio_H, ratio_W, ratio_D


#Asign first photo as -1
def prob_1_bbox():
    first_line = '-1'
    
    for lic_bb in range(photos[0].box_count):
        first_line = '-1'
        
    print(first_line)


#Graph based on: https://pgmpy.org/models/factorgraph.html
def probability():
    result_p = []
    flag = False

    #iterate over all images (except the first one)
    for x, bb in enumerate(photos[1:]):
        Graph = FactorGraph()
        for box in range(bb.box_count):
            photo_name = bb.name + '_' + str(box)
            Graph.add_node(photo_name) 
            for bb_minus_1 in range(photos[x].box_count):
            
                hist_prob = compare_hist(x, bb_minus_1, x+1, box)
                p_h, p_w, p_p = compare_dims(x, bb_minus_1, x+1, box)

                pos_sum = (hist_prob + 2 *p_h + p_p + 2 *p_w) / 6
                result_p.append(pos_sum)

            x1 = DiscreteFactor([photo_name], [len(result_p)+1], [[0.5]+result_p])
            Graph.add_factors(x1)
            Graph.add_node(x1)
            Graph.add_edge(photo_name, x1)
            result_p.clear()

            if bb.box_count >1:
                flag = True

        if flag:

            factor = []
            y1 = np.ones((photos[x].box_count+1, photos[x].box_count+1 ))
            y2 = np.eye(photos[x].box_count+1)
            w = y1 - y2
            w[0][0] += 1

            for j in range(bb.box_count):

                photo_name = bb.name + '_' + str(j)
                factor.append(photo_name)
            factor_1 = [x for x in preparation(factor, 2)]

            for j in range(len(factor_1)):
                x2 = DiscreteFactor([factor_1[j][0], factor_1[j][1]], [photos[x].box_count + 1, photos[x].box_count + 1], w)
                Graph.add_factors(x2)
                Graph.add_node(x2)
                Graph.add_edges_from([(factor_1[j][0], x2), (factor_1[j][1], x2)])
            flag = False


        result = None
        belief_propagation= BeliefPropagation(Graph)
        belief_propagation.calibrate()
        map = belief_propagation.map_query(Graph.get_variable_nodes(), show_progress=False)
        for i in range(bb.box_count):  #assigned bboxes are saved to be printed 
            photo_name = bb.name + '_' + str(i)
            map_info = map[photo_name] - 1

            if not result:
                result = str(map_info)
            else:
                result = result + ' ' + str(map_info)
        print(result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)

    read(images_dir)
    prob_1_bbox()
    probability()
















