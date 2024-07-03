import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color, transform
import joblib
import os


def load_images_from_folder(folder, label, image_size=(28, 28)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder, filename), as_gray=True)
        # cv2.imshow("test2", img);  cv2.waitKey(100)
        if img is not None:
            img = transform.resize(img, image_size)
            # cv2.imshow("test23", img);  cv2.waitKey(100)

            img = img.flatten()
            images.append(img)
            labels.append(label)
    return images, labels

def predict_shape(image_path, model, image_size=(28, 28)):
    img = io.imread(image_path, as_gray=True)
    img = transform.resize(img, image_size)
    img = img.flatten().reshape(1, -1)
    prediction = model.predict(img)
    return prediction


if __name__ == "__main__":
    current_working_directory = os.getcwd()
    print(f"Current working directory: {current_working_directory}")
    
    ref_path = 'neural_network_test/new/cropped2/'
    images_1, labels_1 = load_images_from_folder(ref_path + 'object1', label=1)
    images_2, labels_2 = load_images_from_folder(ref_path + 'object2', label=2)
    images_3, labels_3 = load_images_from_folder(ref_path + 'object3', label=3)
    images_4, labels_4 = load_images_from_folder(ref_path + 'object4', label=4)
    images_5, labels_5 = load_images_from_folder(ref_path + 'object5', label=5)

    X = np.array(images_1 + images_2 + images_3 + images_4 + images_5)
    y = np.array(labels_1 + labels_2 + labels_3 + labels_4 + labels_5)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))



    # # Example usage:
    # pred = predict_shape('neural_network_test/new/cropped2/test3.png', mlp)
    # shape_labels = ['None', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']
    # print("Predicted shape:", shape_labels[pred[0]])


    model_file_path = 'mlp_model.joblib'

    joblib.dump(mlp, model_file_path)
