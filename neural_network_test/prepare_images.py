import numpy as np
import os
import cv2

def load_camera_params(path):
    with np.load(path) as file:
        mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
    return mtx, distortion

def undistort_frame(frame, mtx, distortion):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion, (w, h), 1, (w, h))
    undst = cv2.undistort(frame, mtx, distortion, None, newcameramtx)
    x, y, w, h = roi
    undst = undst[y:y+h, x:x+w]
    return undst


def apply_binarization(gray, binarization='standard'):
    if binarization == 'standard' or binarization == 'raw':
        _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    elif binarization == 'adaptive':
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    elif binarization == 'canny':
        blur = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
        mask = cv2.Canny(blur, 50, 150)
    elif binarization == 'otsu':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Unknown binarization method")
    return mask

def crop_image(image, margin=1, image_raw=None):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]
    cropped_images = []
    cropped_raw_images = []
    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cropped_images.append(masked_image[y:y+h, x:x+w])
        coordinates.append((x, y, w, h))
    if image_raw is not None:
        for contour in contours:
            cropped_raw_images.append(image_raw[y:y+h, x:x+w])
    return cropped_images, coordinates, cropped_raw_images


def prepare_image(image_path, save_path):
    # get camera params
    mtx, distortion = load_camera_params('CameraParams.npz')
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read {image_path}")
        return
    # undistort
    uimg = undistort_frame(img, mtx, distortion)
    # binarize
    buimg = apply_binarization(uimg, 'otsu')
    # find contour + crop
    objects, _, _ = crop_image(buimg)
    # save
    if objects:
        cv2.imwrite(save_path, objects[0])
    else:
        print(f"No objects found in {image_path}")


def process_images_in_folder(reference_folder, results_folder):
    for root, dirs, files in os.walk(reference_folder):

        print(f'root: {root}')
        print(f'dirs: {dirs}')
        print(f'files: {files}')

        # Tworzenie analogicznej struktury folderów wewnątrz folderu wynikowego
        relative_path = os.path.relpath(root, reference_folder)
        save_dir = os.path.join(results_folder, relative_path)
        os.makedirs(save_dir, exist_ok=True)
        
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                save_path = os.path.join(save_dir, file)
                prepare_image(file_path, save_path)
                print(f"Processed and saved {file_path} to {save_path}")


if __name__ == "__main__":
    reference_folder = 'neural_network_test/new/refs2/'
    results_folder = 'neural_network_test/new/cropped2/'

    process_images_in_folder(reference_folder, results_folder)
