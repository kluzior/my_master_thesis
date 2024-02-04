"""
import numpy as np
import cv2 as cv
import glob
import pickle



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8,6)
frameSize = (640,480)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 23
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

num = 0
images = glob.glob('CameraCalibration/images/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.imwrite('images_with_chess/img' + str(num) + '.png', img)
        num += 1

        cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print("Camera Calibrated: ", ret)
...
print("\nCamera Matrix: \n", cameraMatrix)
print("\nDistortion Parameters: \n", dist)
...
print("\nRotation Vectors: \n", rvecs)
print("\nTranslation Vectors: \n", tvecs)



# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))


np.savez("CameraParams", cameraMatrix=cameraMatrix, dist=dist)

############## UNDISTORTION #####################################################

img = cv.imread('CameraCalibration/images/img6.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)



# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
"""

# Import bibliotek
##############################################################################
import numpy as np
import cv2 as cv
import glob
import pickle
##############################################################################

# Deklaracja parametrów pomiaru
##############################################################################
chessboardSize = (8,6) # rozmiar szachownicy użytej do kalibracji
frameSize = (640,480) # rozdzielczość obrazu z kamery
size_of_chessboard_squares_mm = 23 # rzeczywista długość boku kratki szachownicy
##############################################################################

# Kryteria zakończenia
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Przygotowanie macierzy ponumerowanych punktów narożników szachownicy
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# Przemnożenie macierzy punktów przez długość boku - powstała siatka zgodna z rzeczywistością
objp = objp * size_of_chessboard_squares_mm

# Deklaracja zmiennych macierzy do przechowania punktów otrzymanych w analizie
objpoints = [] # punkty 3 wymiarowe w rzeczywistości
imgpoints = [] # punkty 2 wymiarowe na płaszczyźnie zdjęcia

num = 0 # deklaracja numeratora

# Import zdjęć z wskazanego folderu do listy
images = glob.glob('my_master_thesis/captured_images/*.png')
print(len(images))

############## Analiza zdjęć ##############
for image in images:
    img = cv.imread(image) # wczytanie zdjęcia
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # konwersja do skali szarości
    # Funkcja odnajdująca współrzędne rogów siatki szachownicy na zdjęciu
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    ############## Dodanie siatki na obraz ##############
    if ret == True:
        objpoints.append(objp) # dodanie siatki o rzeczywistych wymiarach do macierzy punktów 3D
        # Funkcja odnajdująca lokalizacje narożników z dokładnością zdefiniowanych kryteriów
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2) # dodanie siatki współrzędnych narożników do do macierzy punktów na płaszczyźnie
        # Narysuj szachownicę na zdjęciu i wyświetl je
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        # Zapisz zdjęcie z szachownicą do pliku
        cv.imwrite('images_with_chess/img' + str(num) + '.png', img)
        num += 1 # numerator nazwewnictwa
        cv.waitKey(1000)
    ##############
cv.destroyAllWindows()
##############

############## Kalibracja ##############
# Na podstawie macierzy punktów rzeczywistych i punktów na płaszczyźnie zdjęć jesteśmy w stanie przeprowadzić kalibracje.
# Funkcja zwraca macierz kamery, współczynniki dystorsji oraz wektory rotacji i translacji
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)


# Zapis wyników kalibracji do plików
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))

############## Usunięcie dystorsji ##############
img = cv.imread('my_master_thesis/captured_images/img0.png') # wczytanie obrazu do obróbki
h, w = img.shape[:2] # odczytanie długości (h) i szerokości (w) analizowanego obrazu

# Funkcja obliczająca nową wewnętrzną macierz kamery oraz ROI na podstawie współczynników dystorsji i wymiarów analizowanego obrazu
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


# Usunięcie dystorsji (1 metoda)
# Funkcja dedykowana do usunięcia dystorsji z obrazu dostarczona przez OpenCV
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# Wyciągnięcie współrzędnych z ROI (x,y) - wsp. lewego górnego rogu, (h, w) - długość, szerokość prostokąta
x, y, w, h = roi
dst = dst[y:y+h, x:x+w] # obcięcie obrobionego obrazu do wymiarów ROI
cv.imwrite('undistorted_images/caliResult1.png', dst)


# Usunięcie dystorsji (2 metoda)
# Funkcja obliczająca transformację zniekształceń i rektyfikacji
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None,
newCameraMatrix, (w,h), 5)
# Zastosowanie geometrycznej transformacji obrazu zgodnie z mapami obliczonymi poprzednią funkcją
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# Wyciągnięcie współrzędnych z ROI (x,y) - wsp. lewego górnego rogu, (h, w) - długość, szerokość prostokąta
x, y, w, h = roi
dst = dst[y:y+h, x:x+w] # obcięcie obrobionego obrazu do wymiarów ROI
cv.imwrite('undistorted_images/caliResult2.png', dst)



# Obliczenie błędu ponownego odwzorowania
mean_error = 0 # deklaracja zmiennej błędu
for i in range(len(objpoints)):
    # Przekształcenie punktów 3D obiektu w punkty na płaszczyźnie obrazu
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    # Obliczenie bezwzględnej normy między wynikami transformacji i algorytmu znajdowania narożników
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error # sumowanie błędów
# Wyświetlenie wartości błędu średniego (średnia arytmetyczna błędów każdej kalibracji)
print( "total error: {}".format(mean_error/len(objpoints)) )
# Im błąd reprojekcji jest bliższy zeru, tym dokładniejsze są znalezione parametry.