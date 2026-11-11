from runtime import Accelerator


def run(settings: dict | None = None, accelerated: bool = True, max_epochs: int = 10000) -> dict:

    settings = {} if settings is None else dict(settings)
    accelerator = Accelerator(enabled=accelerated)
    import numpy as np
    import math

    data_len = 100
    np.random.seed(1234)
    data1 = np.arange(0, data_len) / data_len
    data2 = np.sqrt(np.array(data1))
    data = np.hstack((np.array(data1).reshape(-1, 1), np.array(data2).reshape(-1, 1)))
    noise = np.random.normal(0, 0 * (1 - data1**2), len(data1))
    all_y_trues = (
        settings.get("offset", -1) + (data1 + data2) + np.sin(2 * np.pi * (data1 + data2)) + noise
    )
    data = data[np.newaxis, :]


    def mse_loss(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def nearest_child_coordinates(x1, x2, j):
        ind = np.linspace(0, 4 * 2 ** (j - 1), int(4 * 2 ** (j - 1)) + 1)
        x1_w = np.ones((len(x1), 2))
        x2_w = np.ones((len(x1), 2))
        for i in range(len(x1)):
            x1_w[i, :2] = ind[np.where(abs(x1[i] * 2**j - ind) <= 1)][:2]
            x2_w[i, :2] = ind[np.where(abs(x2[i] * 2**j - ind) <= 1)][:2]
        return np.hstack((x1_w, x2_w))

    def unique_child_centers(tr_ce):
        child_centers = np.array([[-100, -100]])
        for i in range(len(tr_ce)):
            for i_1 in [0, 1]:
                for i_2 in [2, 3]:
                    isin = np.any(np.all(tr_ce[i, [i_1, i_2]] == child_centers, axis=1))
                    if isin == 0:
                        child_centers = np.vstack((child_centers, tr_ce[i, [i_1, i_2]]))
        return child_centers[1:, :]


    @accelerator.numpy_class
    class ProbeNetwork:


        def __init__(self, tr_ce, N_1):
            self.w1 = np.random.normal(0, 0, N_1)
            self.k = tr_ce
            shape = self.k.shape
            self.k = self.k.reshape(shape[0], 1, shape[1])

        def sinc(self, x, j, k):
            y = np.sum((2**j * x - k) ** 2, 2) ** 0.5 + np.pi / 1000000.0
            return np.array(
                [2**j * (np.sin(y) / y), 2**j * (-np.sin(2 * y) / y + np.sin(y) / y)]
            )

        def feedforward(self, x, j):
            MH = self.sinc(x, j, self.k)
            return np.dot(MH[1, :, :].T, self.w1)

        def train(self, data, all_y_trues, j, epochs):
            losses = []
            loss_de = 1000
            loss = 100000000.0
            n_sam = len(data[0, :, 0])
            learn_rate = settings.get("learn_rate", 0.0005)
            epoch = 1
            mt = np.zeros_like(self.w1)
            vt = np.zeros_like(self.w1)
            while epoch <= epochs:
                all_y_pred = self.feedforward(data, j)
                activ = self.sinc(data, j, self.k)
                y_L_y_w1_2 = [1 * np.dot(activ[1, :, :], all_y_pred - all_y_trues) / n_sam]
                y_L_y_w1 = np.array(y_L_y_w1_2).reshape(-1)
                mt = 0.9 * mt + (1 - 0.9) * y_L_y_w1
                vt = 0.999 * vt + (1 - 0.999) * y_L_y_w1**2
                self.w1 = self.w1 - learn_rate * mt / (np.sqrt(vt) + 1e-07)
                loss_me = loss
                loss = mse_loss(all_y_trues, all_y_pred)
                losses.append(loss)
                loss_de = abs(loss_me - loss)
                epoch += 1
            return (self.w1, loss)

    tran_r = 2
    dimension = 2
    j_init = settings.get("j_init", 1)
    j = j_init
    epochs = 10
    tr_ce = np.array(
        [
            [i, j_i]
            for i in range(0, int(2 * 2**j + 1), 2)
            for j_i in range(0, int(2 * 2**j + 1), 2)
        ]
    )
    pre_ac = settings.get("pre_ac", 0.005)
    alpha = 2 * np.arctan(-math.log(pre_ac, 10)) / np.pi
    energy = []
    energy_title = []
    reso_ind = 0
    stop_con = 0
    while stop_con == 0 and j <= 8:
        N_1 = len(tr_ce)
        network = ProbeNetwork(tr_ce, N_1)
        (w_need, loss) = network.train(data, all_y_trues, j, epochs)
        tr_ce_1 = tr_ce[:, 0] / 2**j
        tr_ce_2 = tr_ce[:, 1] / 2**j
        tr_ce = nearest_child_coordinates(tr_ce_1, tr_ce_2, j + 1)
        tr_ce = unique_child_centers(tr_ce)
        energy.append(sum(w_need**2))
        energy_title.append(sum(w_need**2))
        if reso_ind > 0:
            energy_title[reso_ind] = (
                alpha * energy_title[reso_ind - 1] + (1 - alpha) * energy[reso_ind]
            ) / (1 - alpha ** (reso_ind + 1))
            if energy[reso_ind] <= energy_title[reso_ind - 1]:
                stop_con = 1
        j = j + 1
        reso_ind = reso_ind + 1
    return locals()
