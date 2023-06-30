import numpy as np
import cv2
import os


def imgScaling(origin_img, scale_factor=0.5):
    img = origin_img.copy()

    height, width = img.shape[:2]
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    scaled_image = cv2.resize(img, (scaled_width, scaled_height))

    top = int((height - scaled_height) / 2)
    bottom = height - scaled_height - top
    left = int((width - scaled_width) / 2)
    right = width - scaled_width - left
    scaled_image = cv2.copyMakeBorder(scaled_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return scaled_image


def imgNoise(origin_img, mean: int = 0, stddev: int = 20):
    img = origin_img.copy()
    noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
    noisy_image = cv2.add(img, noise)
    return noisy_image


def imgColorTrans(origin_img, hue_shift: int = 30, saturation_scale: float = 0.5, value_scale: float = 1.2):
    img = origin_img.copy()
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 调整色调
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 180

    # 调整饱和度
    hsv_image[..., 1] = hsv_image[..., 1] * saturation_scale

    # 调整亮度
    hsv_image[..., 2] = hsv_image[..., 2] * value_scale

    # 将图像转换回BGR颜色空间
    output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return output_image


def dataEnhanced(
        data_path: str = r'',
        save_path: str = r'',
        flip: bool = True,
        rot: bool = True,
        scaling: bool = True,
        noise: bool = True,
        color_trans: bool = True
):
    folder_list = os.listdir(data_path)
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')
    for folder in folder_list:
        file_list = os.listdir(f'{data_path}/{folder}')
        if not os.path.exists(f'{save_path}/{folder}'):
            os.makedirs(f'{save_path}/{folder}')
        for file in file_list:
            img = cv2.imread(f'{data_path}/{folder}/{file}')

            # Save origin
            cv2.imwrite(f'{save_path}/{folder}/{file}', img)

            # Img flip
            if flip:
                flipped_image1 = cv2.flip(img.copy(), 1)
                flipped_image2 = cv2.flip(img.copy(), 0)
                flipped_image3 = cv2.flip(img.copy(), -1)
                cv2.imwrite(f'{save_path}/{folder}/flip1{file}', flipped_image1)
                cv2.imwrite(f'{save_path}/{folder}/flip2{file}', flipped_image2)
                cv2.imwrite(f'{save_path}/{folder}/flip3{file}', flipped_image3)

            # Img rot
            if rot:
                rot1_img = np.rot90(img, -1)
                rot2_img = np.rot90(img, 1)
                cv2.imwrite(f'{save_path}/{folder}/rot1{file}', rot1_img)
                cv2.imwrite(f'{save_path}/{folder}/rot2{file}', rot2_img)

            # Img scaling
            if scaling:
                Scaling9_img = imgScaling(img, 0.9)
                Scaling8_img = imgScaling(img, 0.8)
                Scaling7_img = imgScaling(img, 0.7)
                Scaling6_img = imgScaling(img, 0.5)
                cv2.imwrite(f'{save_path}/{folder}/Scaling9{file}', Scaling9_img)
                cv2.imwrite(f'{save_path}/{folder}/Scaling8{file}', Scaling8_img)
                cv2.imwrite(f'{save_path}/{folder}/Scaling7{file}', Scaling7_img)
                cv2.imwrite(f'{save_path}/{folder}/Scaling6{file}', Scaling6_img)

            # Img noise
            if noise:
                noise1_img = imgNoise(img, mean=0, stddev=20)
                noise2_img = imgNoise(img, mean=0, stddev=30)
                noise3_img = imgNoise(img, mean=10, stddev=30)
                cv2.imwrite(f'{save_path}/{folder}/noise1{file}', noise1_img)
                cv2.imwrite(f'{save_path}/{folder}/noise2{file}', noise2_img)
                cv2.imwrite(f'{save_path}/{folder}/noise3{file}', noise3_img)

            # Img HSV
            if color_trans:
                trans1_img = imgColorTrans(img, hue_shift=30, saturation_scale=0.5, value_scale=1.2)
                trans2_img = imgColorTrans(img, hue_shift=20, saturation_scale=0.3, value_scale=0.9)
                trans3_img = imgColorTrans(img, hue_shift=40, saturation_scale=0.7, value_scale=1.4)
                cv2.imwrite(f'{save_path}/{folder}/trans1{file}', trans1_img)
                cv2.imwrite(f'{save_path}/{folder}/trans2{file}', trans2_img)
                cv2.imwrite(f'{save_path}/{folder}/trans3{file}', trans3_img)


if __name__ == '__main__':
    dataEnhanced(data_path=r'G:\Dataset\Herlev', save_path=r'G:\Dataset\HerlevAug')
