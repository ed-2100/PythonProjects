import time
import matplotlib.axes
import torch
import kornia
import cv2
from apriltag import apriltag

# Example usage
binary_image = torch.tensor([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0]
], dtype=torch.int32)

binary_image = torch.randint(0, 2, (500, 200)).to(device='cpu')

import matplotlib.pyplot as plt

def find_bounding_boxes(labeled_image):
    labels = torch.unique(labeled_image)
    labels = labels[labels != 0]
    bounding_boxes = {}

    for label in labels:
        mask = (labeled_image == label)
        coords = torch.nonzero(mask)
        min_y, min_x = torch.min(coords, dim=0).values
        max_y, max_x = torch.max(coords, dim=0).values

        bounding_boxes[label.item()] = (min_x.item(), min_y.item(), max_x.item(), max_y.item())

    return bounding_boxes

import matplotlib.patches as patches
import matplotlib

def main():
    a = time.time_ns()
    imagepath = 'test_image.png'
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")

    detections = detector.detect(image)
    print(detections, (time.time_ns() - a) / 1e9, sep='\n')

    a = time.time_ns()
    labeled_image = kornia.contrib.connected_components(binary_image.unsqueeze(0).to(dtype=torch.float), 1000).squeeze(0)
    uniques = torch.unique(labeled_image)
    boxes = find_bounding_boxes(labeled_image).values()
    print((time.time_ns() - a) / 1e9)
    
    fig, (ax, ay) = plt.subplots(2)
    ax: matplotlib.axes.Axes
    image = torch.argmax((labeled_image.unsqueeze(-1) == uniques).to(dtype=torch.int8), dim=-1).cpu()
    ax.imshow(image)

    for rect in boxes:
        min_x, min_y, max_x, max_y = rect
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        rect_patch = patches.Rectangle((min_x - 0.5, min_y - 0.5), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_patch)
    
    ax.set_xlim(-0.5, image.shape[1]-0.5)
    ax.set_ylim(image.shape[0]-0.5, -0.5)

    a = time.time_ns()
    components = cv2.connectedComponentsWithStatsWithAlgorithm(binary_image.to(dtype=torch.uint8).numpy(), 8, cv2.CV_16U, cv2.CCL_SAUF)
    print((time.time_ns() - a) / 1e9)
    print(components)

    plt.show()

if __name__=='__main__':
    main()
