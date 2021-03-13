import random
import math


class Lab2:
    variant = 323
    m = 6

    y_max = (30 - variant) * 10
    y_min = (20 - variant) * 10

    x1_min = -5
    x1_max = 15
    x2_min = -25
    x2_max = 10

    xn = [[-1, -1], [1, -1], [-1, 1]]

    def __init__(self):
        self.calculate_and_print()

    @staticmethod
    def average_y(arr):
        average_ny = []
        for i in arr:
            average_ny.append(round(sum(i)/len(i), 2))
        return average_ny

    @staticmethod
    def dispersion(counting_list):
        d = []
        for i in range(len(counting_list)):
            sum_of_y = 0
            for k in counting_list[i]:
                sum_of_y += (k - Lab2.average_y(counting_list)[i]) ** 2
            d.append(round(sum_of_y / len(counting_list[i]), 2))
        return d

    @staticmethod
    def f_uv(u, v):
        if u >= v:
            return u / v
        else:
            return v / u

    @staticmethod
    def determinant(x11, x12, x13, x21, x22, x23, x31, x32, x33):
        det = x11 * x22 * x33 + x12 * x23 * x31 + x32 * x21 * x13 - x13 * x22 * x31 - x32 * x23 * x11 - x12 * x21 * x33
        return det

    @staticmethod
    def theta(m, f):
        return ((m-2)/m)*f

    @staticmethod
    def r(theta, sigma_theta):
        return abs(theta - 1)/sigma_theta

    def calculate_and_print(self):
        y = [[random.randint(self.y_min, self.y_max) for i in range(6)] for j in range(3)]
        print(f'Матриця планування при m = {self.m}')
        for i in range(3):
            print(y[i])

        avg_y = Lab2.average_y(y)
        print(f"\nСереднє значення функції відгуку в рядку (avg_y): {avg_y}")

        print("\nДисперсії по рядках")
        print(f"d(y1): {Lab2.dispersion(y)[0]}")
        print(f"d(y2): {Lab2.dispersion(y)[1]}")
        print(f"d(y3): {Lab2.dispersion(y)[2]}")

        sigma_theta = round(math.sqrt((2 * (2 * self.m - 2)) / (self.m * (self.m - 4))), 2)
        print(f"\nОсновне відхилення: {sigma_theta}\n")

        fuv1 = Lab2.f_uv(Lab2.dispersion(y)[0], Lab2.dispersion(y)[1])
        fuv2 = Lab2.f_uv(Lab2.dispersion(y)[2], Lab2.dispersion(y)[0])
        fuv3 = Lab2.f_uv(Lab2.dispersion(y)[2], Lab2.dispersion(y)[1])

        print(f"Fuv1: {fuv1}")
        print(f"Fuv2: {fuv2}")
        print(f"Fuv3: {fuv3}")

        theta_1 = Lab2.theta(self.m, fuv1)
        theta_2 = Lab2.theta(self.m, fuv2)
        theta_3 = Lab2.theta(self.m, fuv3)

        print(f"\nθuv1: {theta_1}")
        print(f"θuv2: {theta_2}")
        print(f"θuv3: {theta_3}")

        ruv_1 = Lab2.r(theta_1, sigma_theta)
        ruv_2 = Lab2.r(theta_2, sigma_theta)
        ruv_3 = Lab2.r(theta_3, sigma_theta)

        print('Експериментальні значення критерію Романовського:')
        print(f"\nRuv1: {ruv_1}")
        print(f"Ruv2: {ruv_2}")
        print(f"Ruv3: {ruv_3}")

        ruv = [ruv_1, ruv_2, ruv_3]

        r_kr = 2
        for i in range(len(ruv)):
            if ruv[i] > r_kr:
                print("Неоднорідна дисперсія")
                self.m += 1
                return self.calculate_and_print()

        mx1 = (self.xn[0][0] + self.xn[1][0] + self.xn[2][0]) / 3
        mx2 = (self.xn[0][1] + self.xn[1][1] + self.xn[2][1]) / 3
        my = sum(avg_y) / 3

        a1 = (self.xn[0][0] ** 2 + self.xn[1][0] ** 2 + self.xn[2][0] ** 2) / 3
        a2 = (self.xn[0][0] * self.xn[0][1] + self.xn[1][0] * self.xn[1][1] + self.xn[2][0] * self.xn[2][1]) / 3
        a3 = (self.xn[0][1] ** 2 + self.xn[1][1] ** 2 + self.xn[2][1] ** 2) / 3

        a11 = (self.xn[0][0] * avg_y[0] + self.xn[1][0] * avg_y[1] + self.xn[2][0] * avg_y[2]) / 3
        a22 = (self.xn[0][1] * avg_y[0] + self.xn[1][1] * avg_y[1] + self.xn[2][1] * avg_y[2]) / 3

        b0 = Lab2.determinant(my, mx1, mx2, a11, a1, a2, a22, a2, a3) / Lab2.determinant(1, mx1, mx2, mx1, a1, a2, mx2, a2, a3)
        b1 = Lab2.determinant(1, my, mx2, mx1, a11, a2, mx2, a22, a3) / Lab2.determinant(1, mx1, mx2, mx1, a1, a2, mx2, a2, a3)
        b2 = Lab2.determinant(1, mx1, my, mx1, a1, a11, mx2, a2, a22) / Lab2.determinant(1, mx1, mx2, mx1, a1, a2, mx2, a2, a3)

        print("\nНормовані коефіцієнти рівняння регресії:")
        print(f"b0: {b0}")
        print(f"b1: {b1}")
        print(f"b2: {b2}")

        y_pr1 = b0 + b1 * self.xn[0][0] + b2 * self.xn[0][1]
        y_pr2 = b0 + b1 * self.xn[1][0] + b2 * self.xn[1][1]
        y_pr3 = b0 + b1 * self.xn[2][0] + b2 * self.xn[2][1]

        dx1 = abs(self.x1_max - self.x1_min) / 2
        dx2 = abs(self.x2_max - self.x2_min) / 2
        x10 = (self.x1_max + self.x1_min) / 2
        x20 = (self.x2_max + self.x2_min) / 2

        a_0 = b0 - (b1 * x10 / dx1) - (b2 * x20 / dx2)
        a_1 = b1 / dx1
        a_2 = b2 / dx2

        y_p1 = a_0 + a_1 * self.x1_min + a_2 * self.x2_min
        y_p2 = a_0 + a_1 * self.x1_max + a_2 * self.x2_min
        y_p3 = a_0 + a_1 * self.x1_min + a_2 * self.x2_max

        print('\nНатуралізовані коефіцієнти: \na0 =', round(a_0, 4), '\na1 =', round(a_1, 4), '\na2 =', round(a_2, 4))
        print('\nУ практичний: ', round(y_pr1, 4), round(y_pr2, 4), round(y_pr3, 4))
        print('У середній:', round(avg_y[0], 4), round(avg_y[1], 4), round(avg_y[2], 4))
        print('У практичний норм.', round(y_p1, 4), round(y_p2, 4), round(y_p3, 4))


Lab2()
