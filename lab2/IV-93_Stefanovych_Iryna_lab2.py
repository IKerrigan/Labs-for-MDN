import itertools
import numpy as np
from random import *
import math
from functools import *


class Lab2:
    y_max = (30 - 23) * 10
    y_min = (20 - 23) * 10

    x_table = [[-1, -1],
               [-1, +1],
               [+1, -1]]

    p = 0.99

    x1_min = -5
    x1_max = 15
    x2_min = -25
    x2_max = 10

    naturalized_x_table = [[x1_min, x2_min],
                           [x1_min, x2_max],
                           [x1_max, x2_min]]

    def __init__(self):
        self.calculate_and_print()

    def romanovsky_criteria(self, y1: np.array, y2: np.array, y3: np.array):
        def sigma_theta(m):
            return math.sqrt(abs(2 * (2 * m - 2) / (m * (m - 4))))

        def f_uv(y_u: np.array, y_v: np.array):
            dev_u = np.var(y_u)
            dev_v = np.var(y_v)
            return dev_u / dev_v if dev_u > dev_v else dev_v / dev_u

        def theta_uv(m: int, fuv: float):
            return (m - 2) / m * fuv

        def r_uv(s_t: float, s_uv: float):
            return abs(s_uv - 1) / s_t

        def check_criteria(R, m):
            romanovsky_criteria_table = [[None, 2, 6, 8, 10, 12, 15, 20],
                                         [0.99, 1.72, 2.16, 2.43, 2.62, 2.75, 2.90, 3.08],
                                         [0.98, 1.72, 2.13, 2.37, 2.54, 2.66, 2.80, 2.96],
                                         [0.95, 1.71, 2.10, 2.27, 2.41, 2.52, 2.64, 2.78],
                                         [0.90, 1.69, 2.00, 2.17, 2.29, 2.39, 2.49, 2.62]]
            column = romanovsky_criteria_table[0].index(
                sorted(filter(lambda el: el >= m, romanovsky_criteria_table[0][1:]))[0])
            trusted_probability_row = 1
            return R < romanovsky_criteria_table[trusted_probability_row][column]

        sTheta = sigma_theta(self.m)
        accordance = True
        for combination in itertools.combinations((y1, y2, y3), 2):
            fUV = f_uv(combination[0], combination[1])
            sUV = theta_uv(self.m, fUV)
            R = r_uv(sTheta, sUV)
            accordance *= check_criteria(R, self.m)
        return accordance

    def experiment(self):
        return np.array([[randint(self.y_min, self.y_max) for _ in range(self.m)] for _ in range(3)])

    def normalized_regression_coeffs(self):
        def m_i(arr: np.array):
            return np.average(arr)

        def a_i(arr: np.array):
            return sum(arr ** 2) / len(arr)

        def a_jj(arr1: np.array, arr2: np.array):
            return reduce(lambda res, el: res + el[0] * el[1], list(zip(arr1, arr2)), 0) / len(arr1)

        y_vals = np.array([np.average(i) for i in self.y_table])
        x1_vals = np.array([i[0] for i in self.x_table])
        x2_vals = np.array([i[1] for i in self.x_table])
        m_x1 = m_i(x1_vals)
        m_x2 = m_i(x2_vals)
        m_y = m_i(y_vals)
        a1 = a_i(x1_vals)
        a2 = a_jj(x1_vals, x2_vals)
        a3 = a_i(x2_vals)
        a11 = a_jj(x1_vals, y_vals)
        a22 = a_jj(x2_vals, y_vals)
        coeffs_matrix = [[1, m_x1, m_x2],
                         [m_x1, a1, a2],
                         [m_x2, a2, a3]]
        vals_matrix = [m_y, a11, a22]
        b_coeffs = list(map(lambda num: round(num, 2), np.linalg.solve(coeffs_matrix, vals_matrix)))
        self.b_coeffs = b_coeffs
        return b_coeffs

    def assert_normalized_regression(self):
        y_average_experim_vals = np.array([np.average(i) for i in self.y_table])
        print("\nПеревірка правильності знаходження коефіцієнтів рівняння регресії: ")
        print("Середні експериментальні значення y для кожного рядка матриці планування: " +
              ", ".join(map(str, y_average_experim_vals)))
        y_theoretical = [self.b_coeffs[0] + self.x_table[i][0] * self.b_coeffs[1] + self.x_table[i][1] * self.b_coeffs[2] for i in
                         range(len(self.x_table))]
        print("Теоретичні значення y для кожного рядка матриці планування: ".ljust(74) + ", ".join(
            map(str, y_theoretical)))
        for i in range(len(self.x_table)):
            try:
                assert round(y_theoretical[i], 2) == round(y_average_experim_vals[i], 2)
            except:
                print("Неправильні результати пошуку коефіцієнтів рівняння регресії")
                return
        print("Правильні результати пошуку коефіцієнтів рівняння регресії")

    def naturalized_regression(self, b_coeffs: list):
        x1 = abs(self.x1_max - self.x1_min) / 2
        x2 = abs(self.x2_max - self.x2_min) / 2
        x10 = (self.x1_max + self.x1_min) / 2
        x20 = (self.x2_max + self.x2_min) / 2
        a0 = b_coeffs[0] - b_coeffs[1] * x10 / x1 - b_coeffs[2] * x20 / x2
        a1 = b_coeffs[1] / x1
        a2 = b_coeffs[2] / x2
        return [a0, a1, a2]

    def assert_naturalized_regression(self):
        y_average_experim_vals = np.array([np.average(i) for i in self.y_table])
        print("\nПеревірка натуралізації коефіцієнтів рівняння регресії:")
        print("Середні експериментальні значення y для кожного рядка матриці планування: " +
              ", ".join(map(str, y_average_experim_vals)))
        y_theoretical = [self.a_coeffs[0] + self.naturalized_x_table[i][0] * self.a_coeffs[1] + self.naturalized_x_table[i][1] * self.a_coeffs[2]
                         for i in range(len(self.naturalized_x_table))]
        print("Теоретичні значення y для кожного рядка матриці планування: ".ljust(74) + ", ".join(
            map(str, y_theoretical)))
        for i in range(len(self.naturalized_x_table)):
            try:
                assert round(y_theoretical[i], 2) == round(y_average_experim_vals[i], 2)
            except:
                print("Неправильні результати натуралізації")
                return
        print("Правильні результати натуралізації")

    def calculate_and_print(self):
        self.m = 5
        self.y_table = self.experiment()

        while not self.romanovsky_criteria(*self.y_table):
            self.m += 1
            self.y_table = self.experiment()

        labels_table = ["x1", "x2"] + ["y{}".format(i + 1) for i in range(self.m)]
        rows_table = [self.naturalized_x_table[i] + list(self.y_table[i]) for i in range(3)]
        rows_normalized_table = [self.x_table[i] + list(self.y_table[i]) for i in range(3)]

        print("Матриця планування:")
        print((" " * 4).join(labels_table))
        print("\n".join([" ".join(map(lambda j: "{:<+5}".format(j), rows_table[i])) for i in range(len(rows_table))]))
        print("\t")

        print("Нормована матриця планування:")
        print((" " * 4).join(labels_table))
        print("\n".join([" ".join(map(lambda j: "{:<+5}".format(j), rows_normalized_table[i])) for i in
                         range(len(rows_normalized_table))]))
        print("\t")

        b_coeffs = self.normalized_regression_coeffs()
        print("Рівняння регресії для нормованих факторів: y = {0} {1:+}*x1 {2:+}*x2".format(*b_coeffs))
        self.assert_normalized_regression()
        a_coeffs = self.naturalized_regression(b_coeffs)
        self.a_coeffs = a_coeffs
        print("\nРівняння регресії для натуралізованих факторів: y = {0} {1:+}*x1 {2:+}*x2".format(*a_coeffs))
        self.assert_naturalized_regression()

Lab2()




