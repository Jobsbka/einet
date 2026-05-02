"""
exp.py – Эксперименты по спонтанному возникновению геометрической алгебры
в E/I-сети (EinetGA) без прямого заучивания таблицы умножения Cl(3,0).

Эксперименты:
  A : композиция вращений (чётные мультивекторы – кватернионы)
  B : кросс-модальное связывание (вектор × вектор)
  C : восстановление через обратимость (rev(A) × C → B, где C = A·B)
  D : A + B + C (все датасеты вместе)

Метрики:
  L_GA       – MSE на полном базисе геометрического произведения (64 пары)
  Inverter   – MSE двойного отрицания для всех базисных мультивекторов
  Wedge      – MSE для бивекторного произведения e1*e2, e2*e3
  Coincidence – средняя абсолютная ошибка скаляра и бивекторной нормы
               для совпадающих и ортогональных векторов

Для сравнения обучается эталонная сеть на полной таблице умножения.
"""

import random
import math
import os
import sys

# Добавим текущую директорию в путь, если нужно
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from einet_ga import CliffordAlgebra, EinetGA, generate_full_mult_dataset

ga = CliffordAlgebra()

# ------------------------------------------------------------
# Генераторы датасетов
# ------------------------------------------------------------

def generate_dataset_A(num_samples=8000):
    """Пары роторов (кватернионов) и их произведение."""
    dataset = []
    for _ in range(num_samples):
        # случайный единичный кватернион
        q1 = [random.gauss(0,1) for _ in range(4)]
        norm = math.sqrt(sum(x*x for x in q1))
        q1 = [x/norm for x in q1]
        q2 = [random.gauss(0,1) for _ in range(4)]
        norm = math.sqrt(sum(x*x for x in q2))
        q2 = [x/norm for x in q2]

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        # умножение кватернионов
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        # кодируем: [скаляр, 0,0,0, бивекторы x,y,z, 0]
        a = [q1[0], 0.0, 0.0, 0.0, q1[1], q1[2], q1[3], 0.0]
        b = [q2[0], 0.0, 0.0, 0.0, q2[1], q2[2], q2[3], 0.0]
        target = [w, 0.0, 0.0, 0.0, x, y, z, 0.0]
        dataset.append((a, b, target))
    return dataset

def generate_dataset_B(num_samples=8000):
    """Кросс-модальное связывание: только векторы * векторы."""
    dataset = []
    for _ in range(num_samples):
        v1 = [0.0] * 8
        v2 = [0.0] * 8
        # случайные направления в 3D (компоненты 1,2,3)
        dir1 = [random.gauss(0,1) for _ in range(3)]
        norm = math.sqrt(sum(x*x for x in dir1))
        dir1 = [x/norm for x in dir1]
        dir2 = [random.gauss(0,1) for _ in range(3)]
        norm = math.sqrt(sum(x*x for x in dir2))
        dir2 = [x/norm for x in dir2]
        v1[1:4] = dir1
        v2[1:4] = dir2

        target = ga.geometric_product(v1, v2)
        dataset.append((v1, v2, target))
    return dataset

def generate_dataset_C(num_samples=8000):
    """
    Self-supervised восстановление:
    A – случайный вектор, B – случайный полный мультивектор,
    C = A·B, вход = (rev(A), C), цель = B.
    """
    dataset = []
    for _ in range(num_samples):
        # A – единичный вектор
        A = [0.0]*8
        v = [random.gauss(0,1) for _ in range(3)]
        norm = math.sqrt(sum(x*x for x in v))
        A[1:4] = [x/norm for x in v]

        # B – случайный мультивектор
        B = [random.uniform(-1,1) for _ in range(8)]

        # C = A·B
        C = ga.geometric_product(A, B)

        # revA
        revA = ga.reverse(A)

        dataset.append((revA, C, B))
    return dataset

def generate_dataset_D(num_A=3000, num_B=3000, num_C=3000):
    """Объединение A, B, C."""
    ds = []
    ds.extend(generate_dataset_A(num_A))
    ds.extend(generate_dataset_B(num_B))
    ds.extend(generate_dataset_C(num_C))
    return ds

# ------------------------------------------------------------
# Метрики оценки
# ------------------------------------------------------------

def evaluate_ga_error(net):
    """MSE на полном базисе 8x8."""
    basis = [[1.0 if i==j else 0.0 for j in range(8)] for i in range(8)]
    total_se = 0.0
    count = 0
    for a in basis:
        for b in basis:
            pred = net.predict(a, b)
            true = ga.geometric_product(a, b)
            se = sum((pred[k]-true[k])**2 for k in range(8))
            total_se += se
            count += 1
    return total_se / count

def evaluate_inverter(net):
    """MSE двойного отрицания на всех базисных элементах."""
    minus_one = [-1.0, 0,0,0, 0,0,0,0]
    basis = [[1.0 if i==j else 0.0 for j in range(8)] for i in range(8)]
    total_se = 0.0
    for v in basis:
        neg = net.predict(v, minus_one)
        back = net.predict(neg, minus_one)
        se = sum((back[i]-v[i])**2 for i in range(8))
        total_se += se
    return total_se / len(basis)

def evaluate_wedge(net):
    """MSE для e1*e2 и e2*e3 (должны быть чистые бивекторы)."""
    e1 = [0,1,0,0, 0,0,0,0]
    e2 = [0,0,1,0, 0,0,0,0]
    e3 = [0,0,0,1, 0,0,0,0]
    true12 = [0,0,0,0, 1,0,0,0]   # e12
    true23 = [0,0,0,0, 0,0,1,0]   # e23
    pred12 = net.predict(e1, e2)
    pred23 = net.predict(e2, e3)
    se12 = sum((pred12[i]-true12[i])**2 for i in range(8))
    se23 = sum((pred23[i]-true23[i])**2 for i in range(8))
    return (se12 + se23) / 2.0

def evaluate_coincidence(net, num_tests=200):
    """
    Точность детектора совпадений:
    Для пары (v,v) ожидаем скаляр ≈ 1, бивектор ≈ 0.
    Для ортогональной пары (v,u) ожидаем скаляр ≈ 0, норму бивектора ≈ 1.
    Возвращает среднюю абсолютную ошибку по скаляру и бивекторной норме.
    """
    err_scalar = 0.0
    err_binorm = 0.0
    count = 0
    for _ in range(num_tests):
        # единичный вектор v
        v = [0.0]*8
        dir_v = [random.gauss(0,1) for _ in range(3)]
        norm = math.sqrt(sum(x*x for x in dir_v))
        v[1:4] = [x/norm for x in dir_v]

        # совпадающая пара
        out = net.predict(v, v)
        scalar = out[0]
        binorm = math.sqrt(out[4]**2 + out[5]**2 + out[6]**2)
        err_scalar += abs(scalar - 1.0)
        err_binorm += binorm   # ожидаем 0
        count += 1

        # ортогональная пара
        u = [0.0]*8
        # построим вектор, ортогональный v (в 3D)
        if abs(v[1]) > 0.1 or abs(v[2]) > 0.1:
            u[1] = -v[2]
            u[2] = v[1]
            u[3] = 0.0
        else:
            u[1] = 0.0
            u[2] = -v[3]
            u[3] = v[2]
        norm_u = math.sqrt(u[1]**2 + u[2]**2 + u[3]**2)
        u[1] /= norm_u; u[2] /= norm_u; u[3] /= norm_u

        out2 = net.predict(v, u)
        scalar2 = out2[0]
        binorm2 = math.sqrt(out2[4]**2 + out2[5]**2 + out2[6]**2)
        err_scalar += abs(scalar2 - 0.0)
        err_binorm += abs(binorm2 - 1.0)
        count += 1

    return err_scalar / count, err_binorm / count

def evaluate_all(net):
    """Собирает все метрики в словарь."""
    lga = evaluate_ga_error(net)
    inv = evaluate_inverter(net)
    wedge = evaluate_wedge(net)
    sc_err, bi_err = evaluate_coincidence(net)
    return {
        'L_GA': lga,
        'Inverter_MSE': inv,
        'Wedge_MSE': wedge,
        'Coinc_scale_err': sc_err,
        'Coinc_bivec_err': bi_err
    }

# ------------------------------------------------------------
# Обучение и эксперименты
# ------------------------------------------------------------

def train_network(net, dataset, epochs=500, lr=0.05, verbose=False):
    """Обучает сеть на датасете (список (a,b,target))."""
    for epoch in range(epochs):
        total_loss = 0.0
        for a, b, target in dataset:
            total_loss += net.train_step(a, b, target, lr)
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/len(dataset):.6f}")
    return net

def run_experiments():
    print("=== Обучение эталонной сети (полная таблица ГА) ===")
    ref_net = EinetGA()
    full_ds = generate_full_mult_dataset()
    train_network(ref_net, full_ds, epochs=500, lr=0.05, verbose=True)
    ref_metrics = evaluate_all(ref_net)

    print("\n=== Эксперимент A (вращения) ===")
    netA = EinetGA()
    dsA = generate_dataset_A(8000)
    train_network(netA, dsA, epochs=500, lr=0.05, verbose=True)
    metricsA = evaluate_all(netA)

    print("\n=== Эксперимент B (вектор×вектор) ===")
    netB = EinetGA()
    dsB = generate_dataset_B(8000)
    train_network(netB, dsB, epochs=500, lr=0.05, verbose=True)
    metricsB = evaluate_all(netB)

    print("\n=== Эксперимент C (восстановление через rev(A)·C) ===")
    netC = EinetGA()
    dsC = generate_dataset_C(8000)
    train_network(netC, dsC, epochs=500, lr=0.05, verbose=True)
    metricsC = evaluate_all(netC)

    print("\n=== Эксперимент D (A+B+C) ===")
    netD = EinetGA()
    dsD = generate_dataset_D(3000, 3000, 3000)
    train_network(netD, dsD, epochs=500, lr=0.05, verbose=True)
    metricsD = evaluate_all(netD)

    # Вывод сводной таблицы
    print("\n\n===== СВОДНАЯ ТАБЛИЦА МЕТРИК =====")
    header = ["Метрика", "REF (полная)", "A (вращ.)", "B (вект.)", "C (обрат.)", "D (A+B+C)"]
    print("{:<25} {:<14} {:<14} {:<14} {:<14} {:<14}".format(*header))
    print("-" * 85)
    rows = ["L_GA", "Inverter_MSE", "Wedge_MSE", "Coinc_scale_err", "Coinc_bivec_err"]
    for r in rows:
        vals = [ref_metrics[r], metricsA[r], metricsB[r], metricsC[r], metricsD[r]]
        print("{:<25} {:<14.6f} {:<14.6f} {:<14.6f} {:<14.6f} {:<14.6f}".format(r, *vals))

    # Дополнительно: доля E/I весов
    def pos_neg_ratio(net):
        pos, neg, zero = 0, 0, 0
        for row in net.W:
            for w in row:
                if w > 1e-10: pos += 1
                elif w < -1e-10: neg += 1
                else: zero += 1
        total = pos + neg + zero
        return (pos/total, neg/total)
    print("\nДоля положительных/отрицательных весов:")
    for name, net in [("REF", ref_net), ("A", netA), ("B", netB), ("C", netC), ("D", netD)]:
        p, n = pos_neg_ratio(net)
        print(f"  {name}: +{p:.3f}  -{n:.3f}")

if __name__ == "__main__":
    run_experiments()
