import cv2
import numpy as np
import glob

class CameraHandler:
    def __init__(self, camera_params_path):
        self.mtx, self.distortion = self.load_camera_params(camera_params_path)

    def load_camera_params(self, path):
        with np.load(path) as file:
            mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
        return mtx, distortion

    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.distortion, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, self.mtx, self.distortion, None, newcameramtx)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        return undistorted

class ContourDetector:
    def __init__(self, binarization='standard'):
        self.binarization = binarization

    def detect_objects(self, frame, ref_name, min_area=8000, save_mask=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.apply_binarization(gray)
        
        if save_mask:
            cv2.imwrite(f'contour_test/img/detection/mask_{ref_name}.png', mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

    def apply_binarization(self, gray):
        if self.binarization == 'standard':
            _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        elif self.binarization == 'adaptive':
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
        elif self.binarization == 'canny':
            blur = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
            mask = cv2.Canny(blur, 50, 150)
        elif self.binarization == 'otsu':
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError("Unknown binarization method")
        return mask

class ContourMatcher:
    def __init__(self, parent_ref, child_refs, limit_parent=0.25, limit_child=8):
        self.parent_ref = parent_ref
        self.child_refs = child_refs
        self.limit_parent = limit_parent
        self.limit_child = limit_child

    def match_parents(self, contours, parent_idx):
        matched_parent_idx = []
        for parent in parent_idx:
            retval = self.calc_match_shapes(contours[parent], self.parent_ref)
            if retval < self.limit_parent:
                matched_parent_idx.append(parent)
        return matched_parent_idx

    def match_child_of_parent(self, contours, parent_idx, child_dict, img_detection):
        if len(child_dict[parent_idx]) != len(self.child_refs):
            cv2.putText(img_detection, "NUMBER OF DETECTED CHILDREN DON'T MATCH", (200, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            return None
        matched_childs_idx = []
        _child_refs = list(self.child_refs).copy()
        for child in child_dict[parent_idx]:
            min_retval = float('inf')
            best_match_idx = None
            for i, ref in enumerate(_child_refs):
                retval = self.calc_match_shapes(contours[child], ref)
                if retval < min_retval:
                    min_retval = retval
                    best_match_idx = i
            if min_retval < self.limit_child:
                matched_childs_idx.append(child)
                _child_refs.pop(best_match_idx)
                cv2.polylines(img_detection, contours[child], True, (0, 0, 255), 3)
        return matched_childs_idx

    def calc_match_shapes(self, contour, contour_ref):
        retval1 = cv2.matchShapes(contour, contour_ref, cv2.CONTOURS_MATCH_I1, 0)
        retval2 = cv2.matchShapes(contour, contour_ref, cv2.CONTOURS_MATCH_I2, 0)
        retval3 = cv2.matchShapes(contour, contour_ref, cv2.CONTOURS_MATCH_I3, 0)
        mean = (retval1 + retval2 + retval3) / 3
        return mean

class ContourProcessor:
    def __init__(self, camera_params_path, reference_name, binarization_type):
        self.camera_handler = CameraHandler(camera_params_path)
        self.detector = ContourDetector(binarization_type)
        self.reference_name = reference_name
        self.ref_parent, self.ref_childs = self.get_reference(reference_name)
        self.matcher = ContourMatcher(self.ref_parent, self.ref_childs)

    def get_reference(self, ref_name):
        ref = cv2.imread(f"./images/ALL_CONTOURS_2/reference/{ref_name}.png")
        contours, hierarchy = self.detector.detect_objects(ref, ref_name, save_mask=True)
        parent_ref = contours[0] if hierarchy[0][0][3] == -1 else None
        child_refs = contours[1:]
        return parent_ref, child_refs

    def find_parent_index(self, hierarchy):
        return [h for h in range(len(hierarchy)) if hierarchy[h][3] == -1]

    def sort_child_index(self, hierarchy):
        child_dict = {}
        for idx, h in enumerate(hierarchy):
            parent = h[-1]
            if parent == -1:
                continue
            if parent not in child_dict:
                child_dict[parent] = []
            child_dict[parent].append(idx)
        return child_dict

    def process_images(self):
        images = glob.glob('./images/ALL_CONTOURS_2/*.png')
        for num, img_path in enumerate(images):
            img = cv2.imread(img_path)
            uimg = self.camera_handler.undistort_frame(img)
            contours, hierarchy = self.detector.detect_objects(uimg, self.reference_name)
            if hierarchy is None:
                continue
            parents = self.find_parent_index(hierarchy[0])
            childs_sorted = self.sort_child_index(hierarchy[0])
            matched_parents = self.matcher.match_parents(contours, parents)
            img_detection = uimg.copy()
            if matched_parents:
                for matched_parent in matched_parents:
                    for child_idx in childs_sorted[matched_parent]:
                        cv2.polylines(img_detection, contours[child_idx], True, (255, 0, 0), 3)
                    matched_childs = self.matcher.match_child_of_parent(contours, matched_parent, childs_sorted, img_detection)
                    if matched_childs:
                        for child_idx in matched_childs:
                            cv2.polylines(img_detection, contours[child_idx], True, (0, 255, 0), 3)
                for parent_idx in matched_parents:
                    cv2.polylines(img_detection, contours[parent_idx], True, (0, 255, 0), 3)
            cv2.putText(img_detection, f"shape: {self.reference_name}", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img_detection, f"binarization: {self.detector.binarization}", (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.imwrite(f'contour_test/img/detection/{num}_{self.reference_name}_{self.detector.binarization}.png', img_detection)
            cv2.imshow('img_detection', img_detection)
            cv2.waitKey(100)

if __name__ == "__main__":
    reference_names = ['ref1', 'ref2', 'ref3', 'ref4']
    binarization_types = ['standard']
    for reference_name in reference_names:
        for binarization_type in binarization_types:
            processor = ContourProcessor('CameraParams.npz', reference_name, binarization_type)
            processor.process_images()
