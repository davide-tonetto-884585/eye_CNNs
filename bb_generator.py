import os
import cv2


def get_all_files_in_subfolders(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


# Use the function
for file in get_all_files_in_subfolders('./dataset/CASIA-Iris-Distance'):
    print(file)

    # Load the image and convert it to grayscale
    if '.jpg' not in file:
        continue

    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Perform eye detection
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(450, 450), maxSize=(800, 800))

    if len(eyes) == 2:
        # store the eye coordinates in a ini file
        path = os.path.dirname(file)
        filename = os.path.basename(file).split('.')[0]

        with open(f'{path}/{filename}.ini', 'w') as f:
            f.write(f'[eyes]\n')

            f.write(f'x1={eyes[0][0]}\n')
            f.write(f'y1={eyes[0][1]}\n')
            f.write(f'w1={eyes[0][2]}\n')
            f.write(f'h1={eyes[0][3]}\n')

            f.write(f'x2={eyes[1][0]}\n')
            f.write(f'y2={eyes[1][1]}\n')
            f.write(f'w2={eyes[1][2]}\n')
            f.write(f'h2={eyes[1][3]}\n')

    if len(eyes) != 2:
        # delete the image if the eyes are not detected
        os.remove(file)
        print(f'{file} removed')

        break