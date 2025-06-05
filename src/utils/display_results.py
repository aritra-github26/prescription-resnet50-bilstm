import cv2
import numpy as np
from data import preproc as pp

def show_predictions(imgs, gt, predicts, max_display=10):
    """
    Display images with their ground truth and predicted text.

    Args:
        imgs: List of image tensors (C, H, W).
        gt: List of ground truth strings.
        predicts: List of predicted strings.
        max_display: Maximum number of images to display.
    """
    for i, item in enumerate(imgs[:max_display]):
        print("=" * 80)
        img = item.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # Convert to grayscale if image has 3 channels
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = pp.adjust_to_see(img)
        cv2.imshow('Line', img)
        print("Ground truth:", gt[i])
        print("Prediction :", predicts[i], "\n")
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# # To Run:
# from src.utils.display_results import show_predictions

# # Assuming you have run the test function and obtained these:
# predicts, gt, imgs = test(model, test_loader, max_text_length)

# # Call the display function to show images with predictions and ground truth
# show_predictions(imgs, gt, predicts, max_display=10)
