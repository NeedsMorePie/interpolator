import matplotlib.pyplot as plt


def show_image(img):
    """
    Opens a window displaying the image.
    :param img: Numpy array of shape (Height, Width, Channels)
    :return: Nothing.
    """
    plt.imshow(img)
    plt.show()
