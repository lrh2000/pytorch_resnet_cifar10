import torch
import numpy as np
from scipy import linalg
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = ['\\usepackage{amsmath}']

# change with your file path
data_filename = "/tmp/model_tests.10000.pt" 

# https://arxiv.org/abs/1810.11750
# Maximum Matching Similarity
def calc_similarity1(x, y, epsilon):
    x = x.numpy()
    y = y.numpy()
    nr_cols = (x.shape[1] + y.shape[1])

    while True:
        y_basis = linalg.orth(y)
        x_cols = []
        for i in range(x.shape[1]):
            vec = x[:, i:i+1]
            norm = np.sqrt((vec ** 2).sum())
            if norm <= np.finfo(type(norm)).eps * x.max() * x.shape[0]:
                continue
            vec = vec / norm
            weights = np.matmul(vec.T, y_basis)
            proj = np.matmul(y_basis, weights.T)
            remain = vec - proj
            norm = np.sqrt((remain ** 2).sum())
            if norm >= epsilon:
                x_cols.append(i)

        x_basis = linalg.orth(x)
        y_cols = []
        for j in range(y.shape[1]):
            vec = y[:, j:j+1]
            norm = np.sqrt((vec ** 2).sum())
            if norm <= np.finfo(type(norm)).eps * y.max() * y.shape[0]:
                continue
            vec = vec / norm
            weights = np.matmul(vec.T, x_basis)
            proj = np.matmul(x_basis, weights.T)
            remain = vec - proj
            norm = np.sqrt((remain ** 2).sum())
            if norm >= epsilon:
                y_cols.append(j)

        x = np.delete(x, x_cols, axis=1)
        y = np.delete(y, y_cols, axis=1)
        if not x_cols and not y_cols:
            break
        if not x.shape[1] or not y.shape[1]:
            break

    return (x.shape[1] + y.shape[1]) / nr_cols

# https://arxiv.org/abs/1905.00414
# Linear CKA
def calc_similarity2(x, y):
    a = torch.matmul(y.t(), x)
    b = torch.matmul(x.t(), x)
    c = torch.matmul(y.t(), y)
    a = (a ** 2).sum()
    b = (b ** 2).sum().sqrt()
    c = (c ** 2).sum().sqrt()
    z = (a / (b * c)).item()
    return z

def local_perm_similarity(x, y):
    sum1 = 0
    sum2 = 0
    x = sorted((x - min(x)) / (max(x) - min(x)))
    y = sorted((y - min(y)) / (max(y) - min(y)))
    lx = len(x)
    ly = len(y)
    if lx < ly:
        for i in range(ly):
            sum1 = sum1 + min(x[(i * lx) // ly], y[i])
            sum2 = sum2 + max(x[(i * lx) // ly], y[i])
    else:
        for i in range(lx):
            sum1 = sum1 + min(x[i], y[(i * ly) // lx])
            sum2 = sum2 + max(x[i], y[(i * ly) // lx])
    return sum1 / sum2

def universal_perm_similarity(x, y, figure_dir=None, loss_data_limit=1000, plot_data_limit=3):
    if x.shape[1] > y.shape[1]:
        perm = torch.randperm(x.shape[1])
        perm = perm[:y.shape[1]]
        x = x[:, perm]
    elif x.shape[1] < y.shape[1]:
        perm = torch.randperm(y.shape[1])
        perm = perm[:x.shape[1]]
        y = y[:, perm]

    delta = np.zeros((x.size(1), y.size(1)))
    for i in range(min(x.size(0), loss_data_limit) if loss_data_limit else x.size(0)):
        p = x[i, :].numpy()
        q = y[i, :].numpy()
        p = np.tile(p, (y.size(1), 1)).transpose()
        q = np.tile(q, (x.size(1), 1))
        delta += np.abs(p - q) # (np.maximum(p, q) >= 2 * np.minimum(p, q))
        print(i)

    idx, idy = optimize.linear_sum_assignment(delta)
    id_map = [-1] * y.size(1)
    for k in range(len(idx)):
        id_map[idy[k]] = idx[k]

    id_map2 = np.random.permutation(y.size(1))

    result = []
    for i in range(x.size(0)):
        if i == 0:
            p = x.sum(dim=0)
            q0 = y.sum(dim=0)
        elif i <= plot_data_limit:
            p = list(x[i, :].numpy())
            q0 = list(y[i, :].numpy())
        else:
            break

        q = [-1] * len(q0)
        for j in range(len(q0)):
            q[id_map[j]] = q0[j]
        q2 = [-1] * len(q0)
        for j in range(len(q0)):
            q2[id_map2[j]] = q0[j]
        err_q = sorted([abs(p[j] - q[j]) / max([abs(p[j]), abs(q[j]), 0.001]) for j in range(y.size(1))])
        err_q2 = sorted([abs(p[j] - q2[j]) / max([abs(p[j]), abs(q2[j]), 0.001]) for j in range(y.size(1))])
        result += [[sum([1 if abs(err) <= 0.5 else 0 for err in err]) / len(err) for err in [err_q, err_q2]]]

        plt_y = [(i + 1) / len(err_q) for i in range(len(err_q))] + [1.0]
        plt.plot(err_q + [1.0], plt_y, label='$f_\\text{opt}$')
        plt.plot(err_q2 + [1.0], plt_y, label='$\\tilde{f}$')
        plt.xlabel('$w$')
        plt.ylabel('$[\\boldsymbol{E}_r(\\boldsymbol{v}_i,f(\\boldsymbol{u}_i)) \\leq w ]_\\%$')
        plt.legend()
        if figure_dir is not None:
            plt.savefig(os.path.join(figure_dir, f'Figure_{i}a.png'))
            plt.clf()
        else:
            plt.show()

        p, q = zip(*sorted(zip(p, q)))
        plt.plot(q, label='$\\boldsymbol{v}$')
        plt.plot(p, label='$f_\\text{opt}(\\boldsymbol{u})$')
        plt.xlabel('$j$')
        plt.ylabel('$v_{i,j}$/$u_{i,f_\\text{opt}(j)}$')
        plt.legend()
        if figure_dir is not None:
            plt.savefig(os.path.join(figure_dir, f'Figure_{i}b.png'))
            plt.clf()
        else:
            plt.show()

    return (result[0], result[1:])

def main():
    X, Y = torch.load(data_filename)

    X_sum = [x.sum(dim=0) for x in X]
    Y_sum = [y.sum(dim=0) for y in Y]

    print(universal_perm_similarity(X[4], Y[4]))

    for x, y in zip(X_sum, Y_sum):
        plt.plot(sorted(list(x.numpy())))
        plt.plot(sorted(list(y.numpy())))
        plt.show()

        sim = local_perm_similarity(x.numpy(), y.numpy())
        print(f"Similarity: {sim}")
    print('')

    num = 6      # linear 6 cnn 4 resnet 6
    draw_x = np.arange(0, num+1)
    draw_y1 = np.zeros(num)
    draw_y2 = np.zeros(num)
    i = 0
    for x, y in zip(X, Y):
        print(f"x.shape = {x.shape}, y.shape = {y.shape}")
        sim1 = calc_similarity1(x, y, 0.3)
        sim2 = calc_similarity2(x, y)
        draw_y1[i] = sim1
        draw_y2[i] = sim2
        i += 1
        print(f"Similarity1: {sim1}\tSimilarity2: {sim2}")
        draw_y1[i] = sim1
        draw_y2[i] = sim2
        i += 1
        print('')

    # MMS
    plt.xlabel("Layer Number")
    plt.ylabel("Similarity")
    plt.plot(draw_x, draw_y1)
    plt.show()
    # CKA
    plt.xlabel("Layer Number")
    plt.ylabel("Similarity")
    plt.plot(draw_x, draw_y2)
    plt.show()

if __name__ == "__main__":
    main()
