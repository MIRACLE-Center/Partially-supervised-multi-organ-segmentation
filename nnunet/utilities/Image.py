from PIL import Image
import numpy as np
import cv2
import os
from nnunet.paths import *

color_map = {
    'liver': (0, 0, 170),
    'spleen': (0, 170, 170),
    'pancreas': (0, 170, 0),
    'leftkidney': (170, 170, 0),
    'rightkidney': (170, 0, 0)
}

edge_map = {
    'liver': (0, 0, 255),
    'spleen': (0, 255, 255),
    'pancreas': (0, 255, 0),
    'leftkidney': (255, 255, 0),
    'rightkidney': (255, 0, 0)
}

delete_map = {
    'liver': (1, 1, 0),
    'spleen': (1, 0, 0),
    'pancreas': (1, 0, 1),
    'leftkidney': (0, 0, 1),
    'rightkidney': (0, 1, 1)
}


def save(im, save_path=r'/home1/glshi/debug/debugIM.png'):
    if not type(im) is np.ndarray:
        im = np.array(im)
    im = (im-im.min())/(im.max()-im.min())*255
    im = np.uint8(im)
    im = Image.fromarray(np.uint8(im))
    im.save(save_path, 'png')


def save_compare_with_mask_edge(image, prediction, label, label_offset, save_dir, name, predtags, labeltags):
    labeltags = [t.lower() for t in labeltags]
    predtags = [t.lower() for t in predtags]
    if label_offset is None:
        for i, t in enumerate(predtags):
            pred_offset = i+1
            pre = np.where(prediction == pred_offset, 1, 0).astype('uint8')
            image = draw_mask_on_image_cv2(
                image, pre, color=color_map[t])
            if t in labeltags:
                label_offset = labeltags.index(t)+1
                gt = np.where(label == label_offset, 1, 0).astype('uint8')
                image = draw_mask_edge_on_image_cv2(
                    image, gt, color=edge_map[t])
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_path = os.path.join(save_dir, name)
        assert cv2.imwrite(save_path, image)
        return True
    else:
        pre = np.where(prediction == label_offset, 1, 0).astype('uint8')
        gt = np.where(label == label_offset, 1, 0).astype('uint8')
        image = draw_mask_edge_on_image_cv2(
            image, gt, color=edge_map[tag])
        image = draw_mask_on_image_cv2(
            image, pre, color=color_map[tag])
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, name)
        assert cv2.imwrite(save_path, image)
        return True


def save_no_compare(image, prediction, label_offset, save_dir, name, tags):
    if label_offset is None:
        for i, t in enumerate(tags):
            tag = t.lower()
            label_offset = i+1
            pre = np.where(prediction == label_offset, 1, 0).astype('uint8')
            image = draw_mask_on_image_cv2(
                image, pre, color=color_map[tag])
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, name)
        assert cv2.imwrite(save_path, image)
        return True


def WritePredictionWithLabel(image, label, pred, im_name, prediciton_classset, label_classset, z=None):
    print(f"Writing image set:{im_name}")
    image = (image-image.min())/(image.max()-image.min())
    root = image_validation_output_dir
    print(f'shape:{image.shape}/{label.shape}/{pred.shape}')
    if not z is None:
        save_compare_with_mask_edge(image[z, :, :], pred[z, :, :], label[z, :, :], label_offset=None,
                                    save_dir=f'{root}/{im_name}/', name=f'Slice{z:03}.png', predtags=prediciton_classset, labeltags=label_classset)
    else:
        z = label.shape[0]
        for i in range(z):
            save_compare_with_mask_edge(image[i, :, :], pred[i, :, :], label[i, :, :], label_offset=None,
                                        save_dir=f'{root}/{im_name}/', name=f'Slice{i:03}.png', predtags=prediciton_classset, labeltags=label_classset)
        print('save success!')

def WritePrediction(image, pred, im_name, prediciton_classset, z=None):
    print(f"Writing image set:{im_name}")
    image = (image-image.min())/(image.max()-image.min())
    root = image_validation_output_dir
    print(f'shape:{image.shape}/{pred.shape}')
    if not z is None:
        save_no_compare(image[z, :, :], pred[z, :, :],label_offset=None,
                                    save_dir=f'{root}/{im_name}/', name=f'Slice{z:03}.png', tags=prediciton_classset)
    else:
        z = pred.shape[0]
        for i in range(z):
            save_no_compare(image[i, :, :], pred[i, :, :], label_offset=None,
                                        save_dir=f'{root}/{im_name}/', name=f'Slice{i:03}.png', tags=prediciton_classset)



def WritePatchPrediciton(im, prediction, im_name, patch_pos, prediciton_classset=['liver', 'spleen', 'pancreas', 'leftkidney', 'rightkidney'], cpu=True):
    if not cpu:
        im = im.cpu()
        prediction = prediction.cpu()
    root = image_validation_output_dir
    img = np.array(im)
    img = (img-img.min())/(img.max()-img.min())
    p = np.array(prediction)
    p = np.argmax(p, axis=0)
    z = p.shape[0]
    for i in range(z):
        save_no_compare(img[i, :, :], p[i, :, :], label_offset=None,
                        save_dir=f'{root}/{im_name}/{patch_pos}', name=f'Slice{i:03}.png', tags=prediciton_classset)
    print('save success!')


def save_mask_with_image(image, mask, save_root, save_name, class_set):
    for label_offset, name in enumerate(class_set):
        mask_contain = np.where(mask == (label_offset+1), 1, 0).astype('uint8')
        image = draw_mask_on_image_cv2(
            image, mask_contain, color=color_map[name])
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, save_name)
    assert cv2.imwrite(save_path, image)
    return True


def save_mask_with_image_deletechannel(image, mask, save_root, save_name, class_set):
    for label_offset, name in enumerate(class_set):
        mask_contain = np.where(mask == (label_offset+1), 1, 0).astype('uint8')
        image = delete_channel_in_mask(image, mask_contain, delete_map[name])
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, save_name)
    assert cv2.imwrite(save_path, image)
    return True


def draw_mask_edge_on_image_cv2(image, mask, color=(255, 255, 255)):
    coef = 255 if np.max(image) < 3 else 1
    image = (image * coef).astype(np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image, contours, -1, color, thickness=1)
    # cv2.imwrite('test.png', image)
    return image


def save_npz(item, name):
    for i in range(4):
        if len(item.shape) == 5:
            logits = item.cpu().numpy()[0, i+1, :, :, :]
        else:
            logits = item.cpu().numpy()[0, :, :, :]
            logits = np.where(logits == (i+1), 1, 0)
        np.save(f'/home/gongleishi/debug/{i+1}_{name}.npz', logits)


def draw_mask_on_image_cv2(image, mask, alpha=0.6, color=(0, 0, 255)):
    coef = 255 if np.max(image) < 3 else 1
    image = (image * coef).astype(np.float32)
    mask = mask.astype(np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask*color
    mask = np.array(mask).astype(np.float32)
    image = cv2.addWeighted(image, 1, mask, alpha, 0)
    return image


def delete_maskChannel_on_image_cv2(image, mask, alpha=0.6, channel=(0, 1, 1)):
    coef = 255 if np.max(image) < 3 else 1
    image = (image * coef).astype(np.float32)
    mask = mask.astype(np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = delete_channel_in_mask(image, mask, channel)
    return image


def save_single(im, label, im_name, z=None):
    img = im.numpy()
    img = (img-img.min())/(img.max()-img.min())
    la = label.numpy()
    print(
        f'shape:{img.shape}/{la.shape}/{np.max(img)}-{np.min(img)}/{np.max(la)}-{np.min(la)}')
    # im_name = im_name[0].split('/')[7]
    z = la.shape[1]
    for i in range(z):
        for j in range(2):
            save_with_mask_edge_single(
                img[0, i, :, :], la[0, i, :, :], save_root=f'/lustre/T/gongleishi/images_test/{im_name}/{j+1}/', name=f'Slice{i}.png', label_offset=j+1)
    print('save success!')


def save_with_mask_edge_single(image, mask, save_root=r'/home/gongleishi/debug/', name='debugIM.png', label_offset=1):
    mask2 = np.where(mask == label_offset, 1, 0).astype('uint8')
    im_with_mask_edge = draw_mask_edge_on_image_cv2(
        image, mask2, color=(255, 0, 0))
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    save_path = save_root+name
    assert cv2.imwrite(save_path, im_with_mask_edge)
    return True


def draw_ori(im, label, im_name):
    img = im
    print(img.max(), img.min())
    img = (img-img.min())/(img.max()-img.min())
    la = label
    img = img*255
    la = la*128
    z = la.shape[0]
    for i in range(z):
        # save_root = f'/home1/glshi/experiment/img/{im_name}/img'
        # if not os.path.isdir(save_root):
        #     os.makedirs(save_root)
        # save_path = f'/home1/glshi/experiment/img/{im_name}/img/Slice{i:03}.png'
        # a = cv2.cvtColor(img[i, :, :], cv2.COLOR_GRAY2BGR)
        # assert cv2.imwrite(save_path, a)

        save_root = f'/home1/glshi/experiment/img/{im_name}/la/'
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        save_path = f'/home1/glshi/experiment/img/{im_name}/la/Slice{i:03}.png'
        im = draw_mask_on_image_cv2(img[i, :, :], la[i, :, :])
        assert cv2.imwrite(save_path, im)


def delete_channel_in_mask(img, mask, color=(1, 1, 0)):
    # img is BGR 3 channel and mask is [0,1] with 1 channel
    mask = 1-mask
    assert mask.min() >= 0
    mask_render = np.vstack(mask*color[0], mask*color[1], mask*color[2])
    mask_render = np.transpose(mask_render, (1, 2, 0))
    return img*mask_render
