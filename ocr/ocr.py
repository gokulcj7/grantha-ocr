import cv2
import pytesseract
import os
from flask import Flask, render_template, request

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page


@app.route("/")
def home():
    return render_template("home.html")

# Define a route for the OCR page


@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method == "GET":
        return render_template("ocr.html")
    elif request.method == "POST":
        # Get the image file from the request
        image_file = request.files["image"]

        # Save the image to disk
        image_path = os.path.join("static", image_file.filename)
        image_file.save(image_path)

        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the image
        thresholded_image = cv2.threshold(
            grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

        # Find the contours of the image
        contours, hierarchy = cv2.findContours(
            thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the bounding box of the largest contour
        (x, y, w, h) = cv2.boundingRect(largest_contour)

        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Recognize the text in the image
        text = pytesseract.image_to_string(cropped_image, lang="grantha")

        # Return the text to the template
        return render_template("ocr.html", text=text)


if __name__ == "__main__":
    app.run(debug=True)
