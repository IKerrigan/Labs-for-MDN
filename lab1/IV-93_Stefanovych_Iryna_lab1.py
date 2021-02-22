import random
import time


class Lab1:
    start_time = time.time()

    A0 = random.randint(0, 20)
    A1 = random.randint(0, 20)
    A2 = random.randint(0, 20)
    A3 = random.randint(0, 20)

    @staticmethod
    def count_x_0i(x_results) -> int:
        return (max(x_results) + min(x_results)) / 2

    @staticmethod
    def count_dx_i(x0i, x_results) -> int:
        return x0i - min(x_results)

    @staticmethod
    def count_x_ni(x0i, dxi, x_results) -> list:
        return [(i - x0i) / dxi for i in x_results]

    @staticmethod
    def get_optimal_y(y):
        return min(y)

    def __init__(self, n):
        self.N = n  # Number of experiments

        print(f'A0 = {self.A0} \n'
              f'A1 = {self.A1} \n'
              f'A2 = {self.A2} \n'
              f'A3 = {self.A3} \n')

        self.x1, self.x2, self.x3 = [self.generate_x() for i in range(3)]
        self.y = [self.function(self.x1[i], self.x2[i], self.x3[i]) for i in range(8)]

        self.x_01 = Lab1.count_x_0i(self.x1)
        self.x_02 = Lab1.count_x_0i(self.x2)
        self.x_03 = Lab1.count_x_0i(self.x3)

        self.dx_1 = Lab1.count_dx_i(self.x_01, self.x1)
        self.dx_2 = Lab1.count_dx_i(self.x_02, self.x2)
        self.dx_3 = Lab1.count_dx_i(self.x_03, self.x3)

        self.x_1n = Lab1.count_x_ni(self.x_01, self.dx_1, self.x1)
        self.x_2n = Lab1.count_x_ni(self.x_02, self.dx_2, self.x2)
        self.x_3n = Lab1.count_x_ni(self.x_03, self.dx_3, self.x3)

        self.y_et = self.function(self.x_01, self.x_02, self.x_03)

        self.opt_y = Lab1.get_optimal_y(self.y)
        index = self.y.index(self.opt_y)
        self.opt_point = [self.x1[index], self.x2[index], self.x3[index]]

    def function(self, x1, x2, x3):
        return self.A0 + self.A1 * x1 + self.A2 * x2 + self.A3 * x3

    def generate_x(self, start=0, stop=20):
        return [random.randint(start, stop) for _ in range(self.N)]

    def print_formatted_results(self):
        print("N   X1   X2   X3     Y3       XH1    XH2    XH3")
        for i in range(self.N):
            print(f"{i + 1:^1} |{self.x1[i]:^4} {self.x2[i]:^4} {self.x3[i]:^4} |"
                  f" {self.y[i]:^5} || {'%.2f' % self.x_1n[i]:^5}  {'%.2f' % self.x_2n[i]:^5}  {'%.2f' % self.x_3n[i]:^5} |")

        print(f"\nX0| {self.x_01:^4} {self.x_02:^4} {self.x_03:^4}|")
        print(f"dx| {self.dx_1:^4} {self.dx_2:^4} {self.dx_3:^4}|")
        print(f"Function: y = {self.A0} + {self.A1}x1 + {self.A2}x2 + {self.A3}x3")
        print("Yет =", self.y_et)
        print("Optimal point Ymin :  Y({0}, {1}, {2}) = {3}".format(*self.opt_point, "%.1f" % self.opt_y))
        print("Execution time: %s seconds " % (time.time() - self.start_time))


lab = Lab1(8)
lab.print_formatted_results()












