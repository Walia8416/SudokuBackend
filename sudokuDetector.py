





from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
from tensorflow.keras.models import load_model
import imutils
import cv2

def solverNow(location):
    print(location)
    img = cv2.imread(location)

    sudoku_a = cv2.resize(img, (450,450))
    def find_puzzle(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (3,3),20) 
        blur = cv2.bilateralFilter(blur,9,95,95)
        #blur = cv2.GaussianBlur(gray, (3,3), sigmaX=20, sigmaY=20)
        divide = cv2.divide(gray, blur, scale=255)
      
       
        threshold_img = cv2.adaptiveThreshold(divide,255,1,1,11,2)
        
        return threshold_img

    threshold = find_puzzle(sudoku_a)
    
    
    contour_1 = sudoku_a.copy()
    contour_2 = sudoku_a.copy()
    contour, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contour, key = cv2.contourArea, reverse = True)[:5]
    cv2.drawContours(contour_1, contour,-1,(0,255,0),3)
    



    def print_board(board):
        for i in range(len(board)):
            if i % 3 == 0 and i != 0:
                print("------------------------")

            for j in range(len(board[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                if j == 8:
                    print(board[i][j])
                else:
                    print(str(board[i][j]) + " ", end="")

    def find_empty_position(board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)

        return None

    def is_Valid(board, position, number):
        #check row
        for i in range(len(board[0])):
            if board[position[0]][i] == number and position[1] != i:
                return False

        #check column
        for i in range(len(board[0])):
            if board[i][position[1]] == number and position[0] != i:
                return False

        #check cube
        cube_x = position[1] // 3
        cube_y = position[0] // 3

        for i in range(cube_y * 3, cube_y * 3 + 3):
            for j in range(cube_x * 3, cube_x * 3 + 3):
                if board[i][j] == number and (i,j) != position:
                    return False

        return True

    def solve_board(board):
        #print_board(board)
        #print("------------------------")
        empty_position = find_empty_position(board)

        #base case
        if not empty_position:
            return True
        else:
            row, col = empty_position

        for i in range(1, 10):
            if is_Valid(board, (row, col), i):
                board[row][col] = i

                #backtrack and try again
                if solve_board(board):
                    return True
                board[row][col] = 0

        return False








    def main_outline(contour):
        biggest = np.array([])
        max_area = 0
        for i in contour:
            area = cv2.contourArea(i)
            #print(area)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            #print(len(approx))
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
        return biggest ,max_area
    def reframe(points):
        points = points.reshape((4, 2))
        points_new = np.zeros((4,1,2),dtype = np.int32)
        add = points.sum(1)
        points_new[0] = points[np.argmin(add)]
        points_new[3] = points[np.argmax(add)]
        diff = np.diff(points, axis =1)
        points_new[1] = points[np.argmin(diff)]
        points_new[2] = points[np.argmax(diff)]
        return points_new
    def splitcells(img):
        rows = np.vsplit(img,9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r,9)
        
            for box in cols:
                box = cv2.resize(box, (48, 48))
            
                box = box[8:40, 7:40]
                box = cv2.resize(box, (48, 48))
                thresh, box = cv2.threshold(box, 127, 255, cv2.THRESH_BINARY)
                #blur = cv2.GaussianBlur(box, (3,3), sigmaX=52, sigmaY=52)
                divide = cv2.medianBlur(box,15)
                divide = cv2.divide(box, divide, scale=255)
                
                
                boxes.append(divide)
        
    
        return boxes

    black_img = np.zeros((450,450,3), np.uint8)
    biggest, maxArea = main_outline(contour)
    if biggest.size != 0:
        biggest = reframe(biggest)
    cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)

    imagewrap = cv2.warpPerspective(sudoku_a,matrix,(450,450))
    imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
    
    boxes = splitcells(imagewrap)

    rois = np.array(boxes)
    
   
    classes = np.arange(0, 10)
    model = load_model('./model-OCR.h5')
    prediction = model.predict(rois)
    predicted_numbers = []

    for i in prediction:
        index = np.argmax(i)
        predicted_number = classes[int(index)]
        predicted_numbers.append(predicted_number)
    #print(predicted_numbers)

    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)
    #print(board_num)


    solve_board(board_num)
    return board_num

