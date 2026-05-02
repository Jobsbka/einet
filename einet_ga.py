"""
einet_ga.py – E/I-сеть, обучающаяся алгебре Клиффорда Cl(3,0)
без вшитых правил. Используется для построения когнитивных вентилей.
"""

import random
import math
import os

class CliffordAlgebra:
    def __init__(self):
        self.table = self._build_table()

    def _build_table(self):
        T = [[[0.0]*8 for _ in range(8)] for _ in range(8)]
        # Скаляр
        for i in range(8):
            T[0][i][i] = 1.0; T[i][0][i] = 1.0
        # Векторы
        for i in range(1,4): T[i][i][0] = 1.0
        # Вектор × вектор -> бивекторы
        T[1][2][4] = 1.0; T[2][1][4] = -1.0
        T[1][3][5] = 1.0; T[3][1][5] = -1.0
        T[2][3][6] = 1.0; T[3][2][6] = -1.0
        # Бивектор × вектор
        T[4][1][2] = -1.0; T[1][4][2] = 1.0
        T[4][2][1] = 1.0;  T[2][4][1] = -1.0
        T[4][3][7] = 1.0;  T[3][4][7] = 1.0
        T[5][1][3] = -1.0; T[1][5][3] = 1.0
        T[5][2][7] = -1.0; T[2][5][7] = -1.0
        T[5][3][1] = 1.0;  T[3][5][1] = -1.0
        T[6][1][7] = -1.0; T[1][6][7] = 1.0
        T[6][2][3] = -1.0; T[2][6][3] = 1.0
        T[6][3][2] = 1.0;  T[3][6][2] = -1.0
        # Квадраты бивекторов
        T[4][4][0] = -1.0; T[5][5][0] = -1.0; T[6][6][0] = -1.0
        # Бивектор × бивектор
        T[4][5][6] = -1.0; T[5][4][6] = 1.0
        T[4][6][5] = 1.0;  T[6][4][5] = -1.0
        T[5][6][4] = -1.0; T[6][5][4] = 1.0
        # Тривектор
        T[7][7][0] = -1.0
        T[7][1][6] = -1.0; T[1][7][6] = 1.0
        T[7][2][5] = -1.0; T[2][7][5] = 1.0
        T[7][3][4] = 1.0;  T[3][7][4] = 1.0
        T[7][4][3] = -1.0; T[4][7][3] = -1.0
        T[7][5][2] = 1.0;  T[5][7][2] = 1.0
        T[7][6][1] = -1.0; T[6][7][1] = -1.0
        return T

    def geometric_product(self, a, b):
        res = [0.0]*8
        for i in range(8):
            if a[i] == 0.0: continue
            for j in range(8):
                if b[j] == 0.0: continue
                coeff = a[i] * b[j]
                for k in range(8):
                    res[k] += coeff * self.table[i][j][k]
        return res

    def wedge(self, a, b):
        gp = self.geometric_product(a, b)
        return gp[4:7]

    def reverse(self, mv):
        """Правильная реверсия: бивекторы и тривектор меняют знак, векторы — нет."""
        rev = mv[:]
        rev[4] = -rev[4]
        rev[5] = -rev[5]
        rev[6] = -rev[6]
        rev[7] = -rev[7]
        return rev

    def exp_bivector(self, B):
        """
        Экспонента бивектора.
        Норма B интерпретируется как θ/2.
        Возвращает ротор: cos(θ/2) + sin(θ/2)*B̂.
        """
        bx, by, bz = B[4], B[5], B[6]
        norm = math.sqrt(bx*bx + by*by + bz*bz)
        if norm < 1e-8:
            return [1.0, 0,0,0, 0,0,0,0]
        dir_x, dir_y, dir_z = bx/norm, by/norm, bz/norm
        cos_val = math.cos(norm)
        sin_val = math.sin(norm)
        return [cos_val, 0,0,0, sin_val*dir_x, sin_val*dir_y, sin_val*dir_z, 0]

    def apply_rotor(self, R, X):
        return self.geometric_product(
            self.geometric_product(R, X),
            self.reverse(R)
        )

# Глобальный экземпляр алгебры
ga = CliffordAlgebra()

class EinetGA:
    """Линейная сеть для геометрического произведения (64 входа -> 8 выходов)."""
    def __init__(self):
        self.W = [[random.uniform(-0.01, 0.01) for _ in range(8)] for _ in range(64)]

    def _outer_input(self, a, b):
        return [a[i]*b[j] for i in range(8) for j in range(8)]

    def predict(self, a, b):
        inp = self._outer_input(a, b)
        out = [0.0]*8
        for k in range(8):
            s = 0.0
            for idx in range(64):
                s += inp[idx] * self.W[idx][k]
            out[k] = s
        return out

    def train_step(self, a, b, target, lr=0.05):
        inp = self._outer_input(a, b)
        pred = self.predict(a, b)
        errors = [pred[k] - target[k] for k in range(8)]
        for k in range(8):
            for idx in range(64):
                self.W[idx][k] -= lr * errors[k] * inp[idx]
        return sum(e**2 for e in errors)

    def train_on_dataset(self, dataset, epochs=500, lr=0.05, quiet=True):
        for epoch in range(epochs):
            total_loss = 0.0
            for a, b, target in dataset:
                total_loss += self.train_step(a, b, target, lr)
            if not quiet and epoch % 100 == 0:
                print(f"  epoch {epoch}: loss = {total_loss/len(dataset):.6f}")
        return total_loss / len(dataset)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for row in self.W:
                f.write(','.join(str(w) for w in row) + '\n')

    def load(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            self.W = [[float(w) for w in line.strip().split(',')] for line in lines]


class LinearReverser:
    """Линейная модель для операции reverse (8 входов -> 8 выходов)."""
    def __init__(self):
        self.W = [[random.uniform(-0.01, 0.01) for _ in range(8)] for _ in range(8)]

    def predict(self, a):
        out = [0.0]*8
        for k in range(8):
            s = 0.0
            for i in range(8):
                s += a[i] * self.W[i][k]
            out[k] = s
        return out

    def train_step(self, a, target, lr=0.1):
        pred = self.predict(a)
        errors = [pred[k] - target[k] for k in range(8)]
        for k in range(8):
            for i in range(8):
                self.W[i][k] -= lr * errors[k] * a[i]
        return sum(e**2 for e in errors)

    def train_on_dataset(self, dataset, epochs=200, lr=0.1, quiet=True):
        for epoch in range(epochs):
            total_loss = 0.0
            for a, target in dataset:
                total_loss += self.train_step(a, target, lr)
            if not quiet and epoch % 50 == 0:
                print(f"  reverse epoch {epoch}: loss = {total_loss/len(dataset):.6f}")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for row in self.W:
                f.write(','.join(str(w) for w in row) + '\n')

    def load(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            self.W = [[float(w) for w in line.strip().split(',')] for line in lines]


class LinearWedger:
    """Линейная модель для внешнего произведения (64 входа -> 3 выхода)."""
    def __init__(self):
        self.W = [[random.uniform(-0.01, 0.01) for _ in range(3)] for _ in range(64)]

    def predict(self, a, b):
        inp = [a[i]*b[j] for i in range(8) for j in range(8)]
        out = [0.0]*3
        for k in range(3):
            s = 0.0
            for idx in range(64):
                s += inp[idx] * self.W[idx][k]
            out[k] = s
        return out

    def train_step(self, a, b, target_3, lr=0.05):
        inp = [a[i]*b[j] for i in range(8) for j in range(8)]
        pred = self.predict(a, b)
        errors = [pred[k] - target_3[k] for k in range(3)]
        for k in range(3):
            for idx in range(64):
                self.W[idx][k] -= lr * errors[k] * inp[idx]
        return sum(e**2 for e in errors)

    def train_on_dataset(self, dataset, epochs=200, lr=0.05, quiet=True):
        for epoch in range(epochs):
            total_loss = 0.0
            for a, b, target_3 in dataset:
                total_loss += self.train_step(a, b, target_3, lr)
            if not quiet and epoch % 50 == 0:
                print(f"  wedge epoch {epoch}: loss = {total_loss/len(dataset):.6f}")
        return total_loss / len(dataset)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for row in self.W:
                f.write(','.join(str(w) for w in row) + '\n')

    def load(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            self.W = [[float(w) for w in line.strip().split(',')] for line in lines]


def generate_full_mult_dataset():
    """Расширенный датасет для геометрического произведения."""
    basis = [[1 if i==j else 0 for j in range(8)] for i in range(8)]
    dataset = []
    # 1. Базовые пары
    for a in basis:
        for b in basis:
            dataset.append((a, b, ga.geometric_product(a, b)))
    # 2. Критические комбинации
    extras = [
        (basis[4], basis[3]), (basis[5], basis[2]), (basis[6], basis[1]),
        (basis[4], basis[2]), (basis[5], basis[3]), (basis[6], basis[3]),
        (basis[7], basis[1]), (basis[7], basis[2]), (basis[7], basis[3]),
    ]
    for a, b in extras:
        dataset.append((a, b, ga.geometric_product(a, b)))
    # 3. Тривекторные пары
    e123 = basis[7]
    for b in basis:
        dataset.append((e123, b, ga.geometric_product(e123, b)))
        dataset.append((b, e123, ga.geometric_product(b, e123)))
    # 4. Роторные примеры (все векторы во всех плоскостях)
    for X_idx in [1,2,3]:
        X = basis[X_idx]
        for _ in range(150):
            angle = random.uniform(0.1, 2.0)
            plane_idx = random.choice([4,5,6])
            B = [0.0]*8
            B[plane_idx] = 1.0
            R = ga.exp_bivector([(angle/2)*x for x in B])
            R_rev = ga.reverse(R)
            RX = ga.geometric_product(R, X)
            result = ga.geometric_product(RX, R_rev)
            dataset.append((R, X, RX))
            dataset.append((RX, R_rev, result))
            dataset.append((X, R, ga.geometric_product(X, R)))
            dataset.append((R_rev, RX, ga.geometric_product(R_rev, RX)))
    # 5. Случайные примеры
    for _ in range(700):
        a = [random.uniform(-1,1) for _ in range(8)]
        b = [random.uniform(-1,1) for _ in range(8)]
        if random.random() < 0.5:
            a[7] = random.uniform(-1,1)
        else:
            b[7] = random.uniform(-1,1)
        dataset.append((a, b, ga.geometric_product(a, b)))
    return dataset

def generate_reverse_dataset():
    basis = [[1 if i==j else 0 for j in range(8)] for i in range(8)]
    dataset = [(v, ga.reverse(v)) for v in basis]
    for _ in range(20):
        v = [random.uniform(-1,1) for _ in range(8)]
        dataset.append((v, ga.reverse(v)))
    return dataset

def generate_wedge_dataset():
    basis = [[1 if i==j else 0 for j in range(8)] for i in range(8)]
    dataset = []
    for a in basis:
        for b in basis:
            dataset.append((a, b, ga.wedge(a, b)))
    for _ in range(50):
        a = [random.uniform(-1,1) for _ in range(8)]
        b = [random.uniform(-1,1) for _ in range(8)]
        dataset.append((a, b, ga.wedge(a, b)))
    return dataset
