import matplotlib.pyplot as plt


def show_image(image, title: str | None = None) -> None:
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
