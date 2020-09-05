# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
# import tkFileDialog
from tkinter import filedialog
import cv2
import numpy as np

conversion_factor=2.25
binary_threshold=200
min_area=2400
threshold_temperature=100.5

def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature (farenheit) depending upon the camera hardware
    """
    return pixel_avg / conversion_factor


def process_frame(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

    # Binary threshold
    _, binary_thresh = cv2.threshold(
        heatmap_gray, binary_threshold, 255, cv2.THRESH_BINARY)

    # Image opening: Erosion followed by dilation
    kernel = np.ones((5, 5), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    # Get contours from the image obtained by opening operation
    contours, _ = cv2.findContours(image_opening, 1, 2)

    image_with_rectangles = np.copy(heatmap)

    for contour in contours:
        # rectangle over each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Pass if the area of rectangle is not large enough
        if (w) * (h) < min_area:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < threshold_temperature else (
            255, 255, 127)

        # Callback function if the following condition is true
        if temperature >= threshold_temperature:
            # Call back function here
            cv2.putText(image_with_rectangles, "High temperature detected!!!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Draw rectangles for visualisation
        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, (x, y), (x+w, y+h), color, 2)

        # Write temperature for each rectangle
        cv2.putText(image_with_rectangles, "{} F".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return image_with_rectangles



def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()

    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        ori_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        processed_img=process_frame(image)

        # heatmap_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(ori_img)
        edged = Image.fromarray(processed_img)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged


# initialize the window toolkit along with the two image panels
root = Tk()
root.title("Thermal Image Processing")
panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
