import random
import time


class Lab1:
    start_time = time.time()

    a0 = random.randint(0, 20)
    a1 = random.randint(0, 20)
    a2 = random.randint(0, 20)
    a3 = random.randint(0, 20)

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
        self.n = n  # Number of experiments

        print(f'a0 = {self.a0} \n'
              f'a1 = {self.a1} \n'
              f'a2 = {self.a2} \n'
              f'a3 = {self.a3} \n')

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
        return self.a0 + self.a1 * x1 + self.a2 * x2 + self.a3 * x3

    def generate_x(self, start=0, stop=20):
        return [random.randint(start, stop) for _ in range(self.n)]

    def print_formatted_results(self):
        print("n   x1   x2   x3     y3")
        for i in range(self.n):
            print(f"{i + 1:^1} |{self.x1[i]:^4} {self.x2[i]:^4} {self.x3[i]:^4} |"
                  f" {self.y[i]:^5} |")

        print("\n          Factors      ")
        print("n    xh1    xh2    xh3")
        for i in range(self.n):
            print(f"{i + 1:^1} | {'%.2f' % self.x_1n[i]:^5}  {'%.2f' % self.x_2n[i]:^5}  {'%.2f' % self.x_3n[i]:^5} |")

        print(f"\nx0| {self.x_01:^4} {self.x_02:^4} {self.x_03:^4}|")
        print(f"dx| {self.dx_1:^4} {self.dx_2:^4} {self.dx_3:^4}|")
        print(f"\nFunction: y = {self.a0} + {self.a1}x1 + {self.a2}x2 + {self.a3}x3")
        print("Yет =", self.y_et)
        print("Optimal point Ymin :  Y({0}, {1}, {2}) = {3}".format(*self.opt_point, "%.1f" % self.opt_y))
        print("\nExecution time: %s seconds " % (time.time() - self.start_time))


lab = Lab1(8)
lab.print_formatted_results()
