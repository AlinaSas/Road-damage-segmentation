import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np


class Filtres:
    def __init__(self, filtr, image, new_image, **kwargs):
        self.filtr = filtr
        self.image = image
        self.new_image = new_image
        self.params = kwargs
        if filtr == 'black_and_white':
            self.black_white(self.params['bright'])
        if filtr == 'contrast':
            self.contrast(self.params['contrast'])

    def black_white(self, brightness):
        image = Image.open(self.image)
        result = Image.new('RGB', image.size)
        separator = 255 / brightness / 2 * 3
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                r, g, b = image.getpixel((x, y))
                total = r + g + b
                if total > separator:
                    result.putpixel((x, y), (255, 255, 255))
                else:
                    result.putpixel((x, y), (0, 0, 0))
        self.im_save(result)
        return result

    def grayscale(self):
        image = Image.open(self.image)
        pix = image.load()
        draw = ImageDraw.Draw(image)
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                r = pix[x, y][0]
                g = pix[x, y][1]
                b = pix[x, y][2]
                sr = (r + g + b) // 3
                draw.point((x, y), (sr, sr, sr))
        image.save("/home/alina/PycharmProjects/roads/input2/resultB.png", "PNG")

    def inverse(self):
        image = Image.open(self.image)
        pix = image.load()
        draw = ImageDraw.Draw(image)
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                r = pix[x, y][0]
                g = pix[x, y][1]
                b = pix[x, y][2]
                draw.point((x, y), (255 - r, 255 - g, 255 - b))
        image.save("/home/alina/PycharmProjects/roads/input2/result4.png", "PNG")

    def contrast(self, param):
        image = Image.open(self.image)
        enhancer = ImageEnhance.Contrast(image)
        result = enhancer.enhance(param)
        self.im_save(result)

    def rezkost(image_path, result_path, coeff):
        image = Image.open(image_path)
        image = ImageEnhance.Sharpness(image)
        image = image.enhance(coeff)
        image.save(result_path, "PNG")
        # return image

    def contour(self):
        image = Image.open(self.image)
        result = image.filter(ImageFilter.CONTOUR)
        self.im_save(result)
        return result

    def EMBOSS(self):
        image = Image.open(self.image)
        result = image.filter(ImageFilter.EMBOSS)
        self.im_save(result)
        return result

    def FIND_EDGES(self):
        image = Image.open(self.image)
        result = image.filter(ImageFilter.FIND_EDGES)
        self.im_save(result)
        return result

    def gamma_corr(self, coeff):
        img = cv2.imread(self.image)
        result = np.array(255 * (img / 255) ** coeff, dtype='uint8')
        cv2.imwrite(self.new_image, result)

    def im_save(self, im):
        im.save(self.new_image, "PNG")

