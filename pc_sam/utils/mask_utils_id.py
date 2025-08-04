import numpy as np

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def apply_nms(masks, instance_ids, iou_thresh):

    filtered_masks = [] # filtered_masks = [(mask1, id1), (mask2, id2), ...]

    for i in range(len(masks)):
        keep = True

        for j in range(len(filtered_masks)):
            iou = compute_iou(masks[i], filtered_masks[j][0])

            if iou > iou_thresh:
                keep = False
                break

        if keep:
            filtered_masks.append((masks[i], instance_ids[i]))

    return filtered_masks 