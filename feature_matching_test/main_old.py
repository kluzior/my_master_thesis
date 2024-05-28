import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
import logging
import concurrent.futures
from collections import defaultdict

# Configure logging
logging.basicConfig(filename='feature_matching.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def clear_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            logging.warning(f"Path '{directory_path}' does not exist.")
            return
        if not os.path.isdir(directory_path):
            logging.warning(f"Path '{directory_path}' is not a directory.")
            return

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"Removed file: {file_path}")
            except Exception as e:
                logging.error(f"Error removing file '{file_path}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

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


def get_reference(ref_name, binarization, mtx, distortion):
    ref = cv2.imread(f"./feature_matching_test/new/refs/{ref_name}.png", cv2.IMREAD_GRAYSCALE)
    uref = undistort_frame(ref, mtx, distortion)
    uref_bin = apply_binarization(uref, binarization)
    ref_img, ref_coords, raw_img = crop_image(uref_bin, image_raw=uref)
    
    ref_img = raw_img if binarization == 'raw' else ref_img
    return ref_img, ref_coords

def crop_image(image, margin=10, image_raw=None):
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

def process_image(img_path, reference_name, binarization, mtx, distortion, ref_img, ref_coords, results):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    uimg = undistort_frame(img, mtx, distortion)
    uimg_show = uimg.copy()
    # _, uimg = cv2.threshold(uimg, 230, 255, cv2.THRESH_BINARY)
    uimg_bin = apply_binarization(uimg, binarization)

    objects, coordinates, raw_objects = crop_image(uimg_bin, image_raw=uimg)
    objects = raw_objects if binarization == 'raw' else objects

    MIN_MATCH_COUNT = 10
    GOOD_MATCH_RATIO = 0.55
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    all_matches = []

    for obj_img, (x, y, w, h) in zip(objects, coordinates):
        kp1_sift, des1_sift = sift.detectAndCompute(ref_img[0], None)
        kp2_sift, des2_sift = sift.detectAndCompute(obj_img, None)

        if des1_sift is not None and des2_sift is not None:
            matches_sift = flann.knnMatch(des1_sift, des2_sift, k=2)
            good_sift = []
            min_ratio = float('inf')
            max_ratio = 0
            for m, n in matches_sift:
                ratio = m.distance / n.distance
                if ratio < min_ratio:
                    min_ratio = ratio
                if ratio > max_ratio:
                    max_ratio = ratio
                if m.distance < GOOD_MATCH_RATIO * n.distance:
                    good_sift.append(m)

            if len(good_sift) > MIN_MATCH_COUNT:
                for kp in kp2_sift:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                all_matches.append((kp1_sift, kp2_sift, good_sift, x, y, w, h, min_ratio, max_ratio))

    if all_matches:
        all_matches.sort(key=lambda match: len(match[2]), reverse=True)
        for idx, (kp1_sift, kp2_sift, good_sift, x, y, w, h, min_ratio, max_ratio) in enumerate(all_matches):
            img3 = cv2.drawMatches(ref_img[0], kp1_sift, uimg, kp2_sift, good_sift, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.putText(img3, f"REFERENCE NAME: {reference_name}", (0, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img3, f"MIN_MATCH_COUNT : {MIN_MATCH_COUNT}", (0, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img3, f"GOOD_MATCH_RATIO : {GOOD_MATCH_RATIO}", (0, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img3, f"BINARIZATION : {binarization}", (0, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img3, f"Matches found : {len(good_sift)}/{MIN_MATCH_COUNT}", (0, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img3, f"Min ratio: {min_ratio:.2f}", (0, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img3, f"Max ratio: {max_ratio:.2f}", (0, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.imwrite(f'feature_matching_test/results/match_{reference_name}_{binarization}_img_{os.path.basename(img_path)}_match{idx}.png', img3)

        results[reference_name]['total_matches'] += len(all_matches)
        results[reference_name]['match_details'].append({
            'matches': len(all_matches),
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
        })
    else:
        logging.info(f"No matches found above the threshold for {img_path}.")

def run_feature_matching(reference_name, binarization):
    with np.load('CameraParams.npz') as file:
        mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]

    ref_img, ref_coords = get_reference(reference_name, binarization, mtx, distortion)
    images = glob.glob('feature_matching_test/new/*.png')

    results = defaultdict(lambda: {'total_matches': 0, 'match_details': []})

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, img_path, reference_name, binarization, mtx, distortion, ref_img, ref_coords, results) for img_path in images]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # Log results
    logging.info(f"Results for {reference_name}: {results[reference_name]}")

if __name__ == "__main__":
    current_working_directory = os.getcwd()
    logging.info(f"Current working directory: {current_working_directory}")

    directory_path = 'feature_matching_test/results'
    clear_directory(directory_path)

    reference_names = [f'ref{i}_{j}' for i in range(1, 6) for j in range(1, 4)]
    logging.info(f"Reference names: {reference_names}")
    
    binarization_types = ['raw', 'standard', 'adaptive', 'otsu', 'canny']

    for reference_name in reference_names:
        for binarization_type in binarization_types:
            run_feature_matching(reference_name, binarization_type)
