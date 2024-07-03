import numpy as np
import cv2
import glob
import os
import logging
import concurrent.futures
from collections import defaultdict

class FeatureMatcher:
    MIN_MATCH_COUNT = 20
    GOOD_MATCH_RATIO = 0.7

    def __init__(self):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match_features(self, ref_img, obj_img):
        kp1_sift, des1_sift = self.sift.detectAndCompute(ref_img, None)
        kp2_sift, des2_sift = self.sift.detectAndCompute(obj_img, None)

        if des1_sift is None or des2_sift is None:
            return None

        matches_sift = self.flann.knnMatch(des1_sift, des2_sift, k=2)
        # good_sift = [m for m, n in matches_sift if m.distance < self.GOOD_MATCH_RATIO * n.distance]
        good_sift = []
        min_ratio = float('inf')
        max_ratio = 0
        for m, n in matches_sift:
            ratio = m.distance / n.distance
            if ratio < min_ratio:
                min_ratio = ratio
            if ratio > max_ratio:
                max_ratio = ratio
            if m.distance < self.GOOD_MATCH_RATIO * n.distance:
                good_sift.append(m)

        return kp1_sift, kp2_sift, good_sift, min_ratio, max_ratio

    def has_good_match(self, matches):
        return len(matches) > self.MIN_MATCH_COUNT
    

class FeatureProcessor:
    def __init__(self, camera_handler, image_editor, feature_matcher, results_path='feature_matching_test/results'):
        self.camera_handler = camera_handler
        self.image_editor = image_editor
        self.feature_matcher = feature_matcher
        self.results_path = results_path

    def clear_directory(self):
        try:
            if not os.path.exists(self.results_path):
                logging.warning(f"Path '{self.results_path}' does not exist.")
                return
            if not os.path.isdir(self.results_path):
                logging.warning(f"Path '{self.results_path}' is not a directory.")
                return

            for filename in os.listdir(self.results_path):
                file_path = os.path.join(self.results_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logging.info(f"Removed file: {file_path}")
                except Exception as e:
                    logging.error(f"Error removing file '{file_path}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

    def process_image(self, img_path, reference_name, binarization, ref_img, ref_coords, results):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        uimg = self.camera_handler.undistort_frame(img)
        uimg_bin = self.image_editor.apply_binarization(uimg, binarization)

        objects, coordinates, raw_objects = self.image_editor.crop_image(uimg_bin, image_raw=uimg)
        # objects = raw_objects if binarization == 'raw' else objects

        all_matches = []

        for obj_img, (x, y, w, h) in zip(objects, coordinates):
            match_result = self.feature_matcher.match_features(ref_img[0], obj_img)

            if match_result is None:
                continue

            kp1_sift, kp2_sift, good_sift, min_ratio, max_ratio = match_result

            if self.feature_matcher.has_good_match(good_sift):
                for kp in kp2_sift:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                all_matches.append((kp1_sift, kp2_sift, good_sift, x, y, w, h))

        if all_matches:
            all_matches.sort(key=lambda match: len(match[2]), reverse=True)
            for idx, (kp1_sift, kp2_sift, good_sift, x, y, w, h) in enumerate(all_matches):
                img3 = cv2.drawMatches(ref_img[0], kp1_sift, uimg, kp2_sift, good_sift, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.putText(img3, f"REFERENCE NAME: {reference_name}", (0, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img3, f"MIN_MATCH_COUNT : {self.feature_matcher.MIN_MATCH_COUNT}", (0, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img3, f"GOOD_MATCH_RATIO : {self.feature_matcher.GOOD_MATCH_RATIO}", (0, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img3, f"BINARIZATION : {binarization}", (0, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img3, f"Matches found : {len(good_sift)}/{self.feature_matcher.MIN_MATCH_COUNT}", (0, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img3, f"Min ratio: {min_ratio:.2f}", (0, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img3, f"Max ratio: {max_ratio:.2f}", (0, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.imwrite(f'{self.results_path}/match_{reference_name}_img_{os.path.basename(img_path.split(".")[0])}_{binarization}_match{idx}.png', img3)

            results[reference_name]['total_matches'] += len(all_matches)
            results[reference_name]['match_details'].append({
                'matches': len(all_matches),
            })
        else:
            logging.info(f"No matches found above the threshold for {img_path}.")

    def run_feature_matching(self, reference_name, binarization):
        ref_img, ref_coords = self.get_reference(reference_name, binarization)
        images = glob.glob('feature_matching_test/new/*.png')

        results = defaultdict(lambda: {'total_matches': 0, 'match_details': []})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image, img_path, reference_name, binarization, ref_img, ref_coords, results) for img_path in images]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        logging.info(f"Results for {reference_name}: {results[reference_name]}")

    def get_reference(self, ref_name, binarization):
        ref = cv2.imread(f"./feature_matching_test/new/refs/{ref_name}.png", cv2.IMREAD_GRAYSCALE)
        uref = self.camera_handler.undistort_frame(ref)
        uref_bin = self.image_editor.apply_binarization(uref, binarization)
        ref_img, ref_coords, raw_img = self.image_editor.crop_image(uref_bin, image_raw=uref)

        ref_img = raw_img if binarization == 'raw' else ref_img
        return ref_img, ref_coords
