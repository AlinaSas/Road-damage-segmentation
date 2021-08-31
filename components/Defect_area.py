import numpy as np
import math


project_directory = '/home/alina/PycharmProjects/roads/'


class Defect_size:
    """Класс опредления площади повреждения дорожного покрытия.
    Для этого вычисляется пропорция - сколько сантиметров в одном пикселе. Чтобы учесть перспективу,
    изображение делится н заданное количество уровней и на каждом уровне вычисляется пропорция.
    Пропорция вычисляется исходя из того, что известа ширина полосы движения в сантиметрках (375 см),
    известно сколько полос на изображении, и известна ширину отступов и есть сегментированная проезжая часть.
    Учитывается то, что в кадр, на некоторых уровнях, попала дорога не во всю ширину.
    Для этого вычисляются точки краёв проезжей части по уровням.
    Для опредления площади повреждения, считается сколько пикселей составляют повреждение на кждом из уровней,
    и их число умножается на соответсвующее число сантиметров.
    Важно, чтобы повреждение находилось близко к источнику съёмки (по ширине).
    Входные параметры:
    defect_mask_path - путь к изображению с выделенным повреждением,
    road_mask_path - путь к изображению с сегментирванной дорогой. Важно, чтобы съёмка велась максимально близко к
                     проезжей части, чтобы искажения были минимальны.
                     Используется для вычисления пропорции - сколько сантиметров в одном пикселе.
    interval_height - расстояние между уровнми.
    """

    def __init__(self, defect_mask_path, road_mask_path, interval_height=3):
        self.defect_mask = np.load(defect_mask_path)
        self.road_mask = np.load(road_mask_path)

        defect_coord = self.defect_coordinate()
        road_coord = self.road_coordinate()

        R_triangle, L_triangle = self.get_triangles(road_coord)
        K_right, K_left = self.get_continuation_of_the_line(R_triangle, L_triangle, interval_height)
        K_right_newax, K_left_newax = self.switch_to_new_axis(K_right, K_left)
        prop = self.get_proportions(K_right_newax, K_left_newax)
        S = self.get_area(defect_coord, prop)
        print('Площадь повреждения в сантиметрах: ', S)

    def defect_coordinate(self):
        """Определяются коордиаты повреждения"""

        coord = []
        for i in range(len(self.defect_mask)):
            for j in range(len(self.defect_mask[i])):
                if self.defect_mask[i][j][0] == 1:
                    coord.append([i, j])
        return coord

    def road_coordinate(self):
        """Определяются координаты дорожного покрытия"""

        coord_road = []
        for i in range(len(self.road_mask)):
            for j in range(len(self.road_mask[i])):
                if self.road_mask[i][j][0] == 1:
                    coord_road.append([i, j])
        return coord_road

    @staticmethod
    def get_triangles(coord_road):
        """Определяются координаты двух прямоугольных треугольников. ABC - правый треугольник, гипотенуза(AB)
         которого совпадает с правой границей дороги. Координата точки A - точка границы дороги с ординатой 800,
         точка B - точка границы дороги с ординатой, соответсвующей последней попадающей в кадр точке границы дороги,
         точка C - вершина прямоугольного треугольника с гипотенузой AB.
         Треугольник ZYW строится аналогиным образом с левой стороны."""

        BCy, Ay = 352, 352
        YWy, Zy = 352, 352
        for i in range(len(coord_road)):
            if coord_road[i][1] == 1214:
                if coord_road[i][0] < BCy:
                    BCy = coord_road[i][0]

            if coord_road[i][1] == 800:
                if coord_road[i][0] < Ay:
                    Ay = coord_road[i][0]

            if coord_road[i][1] == 1:
                if coord_road[i][0] < YWy:
                    YWy = coord_road[i][0]

            if coord_road[i][1] == 400:
                if coord_road[i][0] < Zy:
                    Zy = coord_road[i][0]

        A, B, C = [800, Ay], [1214, BCy], [800, BCy]
        Z, Y, W = [400, Zy], [0, YWy], [400, YWy]
        return [A, B, C], [Z, Y, W]

    @staticmethod
    def get_continuation_of_the_line(R_triangle, L_triangle, interval_height):
        """Вычисляются точки краёв дорожного покрытия спарва и слева по уровням"""

        A, B, C = R_triangle[0], R_triangle[1], R_triangle[2]
        Z, Y, W = L_triangle[0], L_triangle[1], L_triangle[2]
        AB = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        AC = math.sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)

        ZY = math.sqrt((Z[0] - Y[0]) ** 2 + (Z[1] - Y[1]) ** 2)
        ZW = math.sqrt((Z[0] - W[0]) ** 2 + (Z[1] - W[1]) ** 2)

        cosCAB = AC / AB
        cosYZW = ZW / ZY
        K_right, K_left = [], []
        povorot = False
        for i in range(352 // interval_height):

            P_right = [800, 352 - i * interval_height]
            P_left = [400, 352 - i * interval_height]
            AP_left = math.sqrt((A[0] - P_right[0]) ** 2 + (A[1] - P_right[1]) ** 2)
            ZP_right = math.sqrt((Z[0] - P_left[0]) ** 2 + (Z[1] - P_left[1]) ** 2)

            AK_left = AP_left / cosCAB
            ZK_right = ZP_right / cosYZW

            K01 = int((A[0] + math.sqrt(AK_left ** 2 - (A[1] - (352 - i * interval_height)) ** 2)))
            K02 = int((Z[0] - math.sqrt(ZK_right ** 2 - (Z[1] - (352 - i * interval_height)) ** 2)))
            if AP_left < 5:
                povorot = True
            if povorot:
                K01 = int((A[0] - math.sqrt(AK_left ** 2 - (A[1] - 352 + i * interval_height) ** 2)))
                K02 = int((Z[0] + math.sqrt(ZK_right ** 2 - (Z[1] - 352 + i * interval_height) ** 2)))
            if K01 <= K02:
                K01, K02 = 0, 0
            K_right.append([K01, 352 - i * interval_height])
            K_left.append([K02, 352 - i * interval_height])
        return K_right, K_left

    @staticmethod
    def switch_to_new_axis(K1, K2):
        """Происходит переход к новым осям координат, смещая ноь в левый верхний угол"""

        new_zero = 0
        for i in range(len(K2)):
            for j in range(len(K1[i])):
                if K2[i][0] < new_zero:
                    new_zero = K2[i][0]

        K1N, K2N = [], []
        for k1, k2 in zip(K1, K2):
            K1N.append([k1[0] + abs(new_zero), k1[1]])
            K2N.append([k2[0] + abs(new_zero), k2[1]])
        return K1N, K2N

    @staticmethod
    def get_proportions(K1N, K2N):
        """Вычисляются пропорции, т.е. сколько сантиметров в пикселе по уровням"""
        line_coord = {}
        for k1, k2 in zip(K1N, K2N):
            if k1 != k2:
                line_coord[str(k1[1])] = [k2[0], k1[0]]
        print('Координаты продлённых линий по уровням в смещённых осях: ', line_coord)

        proportion = {}
        for key in line_coord.keys():
            lenght = line_coord[key][1] - line_coord[key][0]
            proportion[key] = (375 * 3 + 40) / lenght

        print('Найденные пропорции(сколько сантиметров в одном пикселе) по уровням: ', proportion)
        return proportion

    @staticmethod
    def get_area(defect_coord, proportion):
        """Вычисляется плозадь повреждения"""

        ll = list(proportion.keys())
        S = 0
        for i in range(len(ll) - 1):
            a = []
            for c in defect_coord:
                if float(ll[i]) >= c[0] > float(ll[i + 1]):
                    a.append(c)
            S += len(a) * proportion[ll[i]]

        return S


d = Defect_size(project_directory + 'imgs/Road_defects/NPY/np_masks/104___video_only_img11241.npy',
                project_directory + 'imgs/Road_surface/NPY/val_np_masks/104___video_only_img135.npy')

