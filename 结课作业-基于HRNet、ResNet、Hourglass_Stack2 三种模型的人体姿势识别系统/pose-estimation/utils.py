import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage
import torch
from PIL import Image

joints = [
    'left ankle',
    'left knee',
    'left hip',
    'right hip',
    'right knee',
    'right ankle',
    'belly',
    'chest',
    'neck',
    'head',
    'left wrist',
    'left elbow',
    'left shoulder',
    'right shoulder',
    'right elbow',
    'right wrist'
]


def generate_heatmap(heatmap, pt, sigma=(33, 33), sigma_valu=7):
    '''
    :param heatmap: should be a np zeros array with shape (H,W) (only i channel), not (H,W,1)
    :param pt: point coords, np array
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: vaalue for gaussian blur
    :return: a np array of one joint heatmap with shape (H,W)

    This function is obsolete, use 'generate_heatmaps()' instead.
    '''
    heatmap[int(pt[1])][int(pt[0])] = 1
    # heatmap = cv2.GaussianBlur(heatmap, sigma, 0)  #(H,W,1) -> (H,W)
    heatmap = skimage.filters.gaussian(
        heatmap, sigma=sigma_valu)  # (H,W,1) -> (H,W)
    am = np.amax(heatmap)
    heatmap = heatmap / am
    return heatmap


def generate_heatmaps(img, pts, sigma=(33, 33), sigma_valu=7):
    '''
    :param img: np arrray img, (H,W,C)
    :param pts: joint points coords, np array, same resolu as img
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: vaalue for gaussian blur
    :return: np array heatmaps, (H,W,num_pts)
    '''
    H, W = img.shape[0], img.shape[1]
    num_pts = pts.shape[0]
    heatmaps = np.zeros((H, W, num_pts))
    for i, pt in enumerate(pts):
        # Filter unavailable heatmaps
        if pt[0] == 0 and pt[1] == 0:
            continue
        # Filter some points out of the image
        if pt[0] >= W:
            pt[0] = W - 1
        if pt[1] >= H:
            pt[1] = H - 1
        heatmap = heatmaps[:, :, i]
        heatmap[int(pt[1])][int(pt[0])] = 1
        # heatmap = cv2.GaussianBlur(heatmap, sigma, 0)  #(H,W,1) -> (H,W)
        heatmap = skimage.filters.gaussian(
            heatmap, sigma=sigma_valu)  # (H,W,1) -> (H,W)
        am = np.amax(heatmap)
        heatmap = heatmap / am
        heatmaps[:, :, i] = heatmap
    return heatmaps


def load_image(path_image):
    img = mpimg.imread(path_image)
    # Return a np array (H,W,C)
    return img


def crop(img, ele_anno, use_randscale=True, use_randflipLR=False, use_randcolor=False):
    '''
    :param img: np array of the origin image, (H,W,C)
    :param ele_anno: one element of json annotation
    :return: img_crop, ary_pts_crop, c_crop after cropping
    '''

    H, W = img.shape[0], img.shape[1]
    s = ele_anno['scale_provided']
    c = ele_anno['objpos']

    # Adjust center and scale
    if c[0] != -1:
        c[1] = c[1] + 15 * s
        s = s * 1.25
    ary_pts = np.array(ele_anno['joint_self'])  # (16, 3)
    ary_pts_temp = ary_pts[np.any(ary_pts != [0, 0, 0], axis=1)]

    if use_randscale:
        scale_rand = np.random.uniform(low=1.0, high=3.0)
    else:
        scale_rand = 1

    W_min = max(np.amin(ary_pts_temp, axis=0)[0] - s * 15 * scale_rand, 0)
    H_min = max(np.amin(ary_pts_temp, axis=0)[1] - s * 15 * scale_rand, 0)
    W_max = min(np.amax(ary_pts_temp, axis=0)[0] + s * 15 * scale_rand, W)
    H_max = min(np.amax(ary_pts_temp, axis=0)[1] + s * 15 * scale_rand, H)
    W_len = W_max - W_min
    H_len = H_max - H_min
    window_len = max(H_len, W_len)
    pad_updown = (window_len - H_len) / 2
    pad_leftright = (window_len - W_len) / 2

    # Calculate 4 corner position
    W_low = max((W_min - pad_leftright), 0)
    W_high = min((W_max + pad_leftright), W)
    H_low = max((H_min - pad_updown), 0)
    H_high = min((H_max + pad_updown), H)

    # Update joint points and center
    ary_pts_crop = np.where(
        ary_pts == [0, 0, 0], ary_pts, ary_pts - np.array([W_low, H_low, 0]))
    c_crop = c - np.array([W_low, H_low])

    img_crop = img[int(H_low):int(H_high), int(W_low):int(W_high), :]

    # Pad when H, W different
    H_new, W_new = img_crop.shape[0], img_crop.shape[1]
    window_len_new = max(H_new, W_new)
    pad_updown_new = int((window_len_new - H_new) / 2)
    pad_leftright_new = int((window_len_new - W_new) / 2)

    # ReUpdate joint points and center (because of the padding)
    ary_pts_crop = np.where(ary_pts_crop == [
        0, 0, 0], ary_pts_crop, ary_pts_crop + np.array([pad_leftright_new, pad_updown_new, 0]))
    c_crop = c_crop + np.array([pad_leftright_new, pad_updown_new])

    img_crop = cv2.copyMakeBorder(img_crop, pad_updown_new, pad_updown_new,
                                  pad_leftright_new, pad_leftright_new, cv2.BORDER_CONSTANT, value=0)

    # change dtype and num scale
    img_crop = img_crop / 255.
    img_crop = img_crop.astype(np.float64)

    if use_randflipLR:
        flip = np.random.random() > 0.5
        # print('rand_flipLR', flip)
        if flip:
            # (H,W,C)
            img_crop = np.flip(img_crop, 1)
            # Calculate flip pts, remember to filter [0,0] which is no available heatmap
            ary_pts_crop = np.where(ary_pts_crop == [0, 0, 0], ary_pts_crop,
                                    [window_len_new, 0, 0] + ary_pts_crop * [-1, 1, 0])
            c_crop = [window_len_new, 0] + c_crop * [-1, 1]
            # Rearrange pts
            ary_pts_crop = np.concatenate(
                (ary_pts_crop[5::-1], ary_pts_crop[6:10], ary_pts_crop[15:9:-1]))

    if use_randcolor:
        randcolor = np.random.random() > 0.5
        # print('rand_color', randcolor)
        if randcolor:
            img_crop[...,
            0] *= np.clip(np.random.uniform(low=0.8, high=1.2), 0., 1.)
            img_crop[...,
            1] *= np.clip(np.random.uniform(low=0.8, high=1.2), 0., 1.)
            img_crop[...,
            2] *= np.clip(np.random.uniform(low=0.8, high=1.2), 0., 1.)

    return img_crop, ary_pts_crop, c_crop


def change_resolu(img, pts, c, resolu_out=(256, 256)):
    '''
    :param img: np array of the origin image
    :param pts: joint points np array corresponding to the image, same resolu as img
    :param c: center
    :param resolu_out: a list or tuple
    :return: img_out, pts_out, c_out under resolu_out
    '''
    H_in = img.shape[0]
    W_in = img.shape[1]
    H_out = resolu_out[0]
    W_out = resolu_out[1]
    H_scale = H_in / H_out
    W_scale = W_in / W_out

    pts_out = pts / np.array([W_scale, H_scale, 1])
    c_out = c / np.array([W_scale, H_scale])
    img_out = skimage.transform.resize(img, tuple(resolu_out))

    return img_out, pts_out, c_out


def heatmaps_to_coords(heatmaps, resolu_out=[64, 64], prob_threshold=0.2):
    '''
    :param heatmaps: tensor with shape (64,64,16)
    :param resolu_out: output resolution list
    :return coord_joints: np array, shape (16,2)
    '''

    num_joints = heatmaps.shape[2]
    # Resize
    heatmaps = skimage.transform.resize(heatmaps, tuple(resolu_out))

    coord_joints = np.zeros((num_joints, 3))
    for i in range(num_joints):
        heatmap = heatmaps[..., i]
        max = np.max(heatmap)
        # Only keep points larger than a threshold
        if max >= prob_threshold:
            idx = np.where(heatmap == max)
            H = idx[0][0]
            W = idx[1][0]
        else:
            H = 0
            W = 0
        coord_joints[i] = [W, H, max]
    return coord_joints


def show_heatmaps(img, heatmaps, c=np.zeros((2)), num_fig=1):
    '''
    :param img: np array (H,W,3)
    :param heatmaps: np array (H,W,num_pts)
    :param c: center, np array (2,)
    '''
    H, W = img.shape[0], img.shape[1]

    if heatmaps.shape[0] != H:
        heatmaps = skimage.transform.resize(heatmaps, (H, W))

    plt.figure(num_fig)
    for i in range(heatmaps.shape[2] + 1):
        plt.subplot(4, 5, i + 1)
        if i == 0:
            plt.title('Origin')
        else:
            plt.title(joints[i - 1])

        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(heatmaps[:, :, i - 1])

        plt.axis('off')
    plt.subplot(4, 5, 20)
    plt.axis('off')
    plt.show()


def heatmap2rgb(heatmap):
    """
    : heatmap: (h,w)
    """

    heatmap = heatmap.detach().cpu().numpy()

    # plt.figure(figsize=(1,1))
    # plt.axis('off')
    # plt.imshow(heatmap)
    # plt.savefig('tmp/tmp.jpg', bbox_inches='tight', pad_inches=0, dpi=70)
    # plt.close()
    # plt.clf()

    # img = Image.open('tmp/tmp.jpg')
    cm = plt.get_cmap('jet')
    normed_data = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap + 1e-8))
    mapped_data = cm(normed_data)

    # (h,w,c)
    # img = np.array(img)
    img = np.array(mapped_data)
    img = img[:, :, :3]
    img = torch.tensor(img).permute(2, 0, 1)

    return img


def heatmaps2rgb(heatmaps):
    """
    : heatmaps: (b,h,w)
    """

    out_imgs = []
    for heatmap in heatmaps:
        out_imgs.append(heatmap2rgb(heatmap))

    return torch.stack(out_imgs)


def draw_joints(img, pts):
    scores = pts[:, 2]
    pts = np.array(pts).astype(int)

    for i in range(pts.shape[0]):
        if pts[i, 0] != 0 and pts[i, 1] != 0:
            img = cv2.circle(img, (pts[i, 0], pts[i, 1]), radius=5,
                             color=(255, 0, 0), thickness=-1)
            img = cv2.putText(img, f'{joints[i]}: {scores[i]:.2f}', (
                pts[i, 0] + 5, pts[i, 1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 0, 0))

    # Left arm
    for i in range(10, 13 - 1):
        if pts[i, 0] != 0 and pts[i, 1] != 0 and pts[i + 1, 0] != 0 and pts[i + 1, 1] != 0:
            img = cv2.line(img, (pts[i, 0], pts[i, 1]), (pts[i + 1, 0],
                                                         pts[i + 1, 1]), color=(255, 0, 0), thickness=1)

    # Right arm
    for i in range(13, 16 - 1):
        if pts[i, 0] != 0 and pts[i, 1] != 0 and pts[i + 1, 0] != 0 and pts[i + 1, 1] != 0:
            img = cv2.line(img, (pts[i, 0], pts[i, 1]), (pts[i + 1, 0],
                                                         pts[i + 1, 1]), color=(255, 0, 0), thickness=1)

    # Left leg
    for i in range(0, 3 - 1):
        if pts[i, 0] != 0 and pts[i, 1] != 0 and pts[i + 1, 0] != 0 and pts[i + 1, 1] != 0:
            img = cv2.line(img, (pts[i, 0], pts[i, 1]), (pts[i + 1, 0],
                                                         pts[i + 1, 1]), color=(255, 0, 0), thickness=1)
    # Right leg
    for i in range(3, 6 - 1):
        if pts[i, 0] != 0 and pts[i, 1] != 0 and pts[i + 1, 0] != 0 and pts[i + 1, 1] != 0:
            img = cv2.line(img, (pts[i, 0], pts[i, 1]), (pts[i + 1, 0],
                                                         pts[i + 1, 1]), color=(255, 0, 0), thickness=1)

    # Body
    for i in range(6, 10 - 1):
        if pts[i, 0] != 0 and pts[i, 1] != 0 and pts[i + 1, 0] != 0 and pts[i + 1, 1] != 0:
            img = cv2.line(img, (pts[i, 0], pts[i, 1]), (pts[i + 1, 0],
                                                         pts[i + 1, 1]), color=(255, 0, 0), thickness=1)

    if pts[2, 0] != 0 and pts[2, 1] != 0 and pts[3, 0] != 0 and pts[3, 1] != 0:
        img = cv2.line(img, (pts[2, 0], pts[2, 1]), (pts[2 + 1, 0],
                                                     pts[2 + 1, 1]), color=(255, 0, 0), thickness=1)
    if pts[12, 0] != 0 and pts[12, 1] != 0 and pts[13, 0] != 0 and pts[13, 1] != 0:
        img = cv2.line(img, (pts[12, 0], pts[12, 1]), (pts[12 + 1, 0],
                                                       pts[12 + 1, 1]), color=(255, 0, 0), thickness=1)

    return img
