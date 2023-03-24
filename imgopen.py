import matplotlib.pyplot as plt

# Load the image file
img_file = "filename.png"
img = Image.open(img_file)

# Display the image using matplotlib
plt.imshow(img)
plt.show()
