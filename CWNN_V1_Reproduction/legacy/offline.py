from runtime import Accelerator


def run(settings: dict | None = None, accelerated: bool = True, max_epochs: int = 10000) -> dict:

    settings = {} if settings is None else dict(settings)
    accelerator = Accelerator(enabled=accelerated)
    import numpy as np
    import time

    seed_set = 1234
    data_len = 100
    np.random.seed(seed_set)
    data1 = np.arange(0, data_len) / data_len
    data2 = np.sqrt(np.array(data1))
    data = np.hstack((np.array(data1).reshape(-1, 1), np.array(data2).reshape(-1, 1)))
    noise = np.random.normal(0, settings.get("noise", 0) * (1 - data1**2), len(data1))
    all_y_trues = (
        settings.get("offset", -1) + (data1 + data2) + np.sin(2 * np.pi * (data1 + data2)) + noise
    )
    data = data[np.newaxis, :]
    np.random.seed(1234)
    n_samples = data.shape[1]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(n_samples * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    data_train = data[:, train_indices, :]
    all_y_trues_train = all_y_trues[train_indices]
    data_test = data[:, test_indices, :]
    all_y_trues_test = all_y_trues[test_indices]


    def mse_loss(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def energy_separate(w_need, mu, j, tran_r, dimension):
        N_J = int((tran_r * 2**j + 1) ** dimension)
        w_weights = w_need[-1 : -(N_J + 1) : -1]
        w_weights1 = np.sort(abs(w_weights))[::-1]
        sum_w1 = np.zeros((len(w_weights1), 1)).reshape(-1)
        sum_w1[0] = w_weights1[0] ** 2
        for i in range(1, len(w_weights1)):
            sum_w1[i] = sum_w1[i - 1] + w_weights1[i] ** 2
        ratio = sum_w1 / sum_w1[-1]
        center_groups = []
        previous_cutoff = 0
        for i_ind in range(1, int(1 / mu) + 1):
            condition_indices = np.where(ratio <= i_ind * mu)[0]
            if condition_indices.size == 0:
                cutoff = 0 + 1
            else:
                cutoff = np.max(condition_indices) + 1
            center_groups.append(np.argsort(abs(w_weights1))[previous_cutoff:cutoff])
            previous_cutoff = cutoff
        return center_groups


    @accelerator.numpy_class
    class InitialNetwork:


        def __init__(self, N_w1):
            self.w1 = np.random.normal(0, 0, N_w1)
            wave_t = int(np.sqrt(N_w1 / 2))
            self.k = np.array([[i, j_i] for i in range(wave_t) for j_i in range(wave_t)])
            shape = self.k.shape
            self.k = self.k.reshape(shape[0], 1, shape[1])

        def sinc(self, x, j, k):
            y = np.sum((2**j * x - k) ** 2, 2) ** 0.5 + np.pi / 1000000.0
            return np.array(
                [2**j * (np.sin(y) / y), 2**j * (-np.sin(2 * y) / y + np.sin(y) / y)]
            )

        def feedforward(self, x, j):
            MH = self.sinc(x, j, self.k)
            return np.dot(np.concatenate((MH[0, :, :], MH[1, :, :]), 0).T, self.w1)

        def train(
            self,
            data,
            all_y_trues,
            j,
            pre_ac,
            pre_term,
            learn_rate,
            data_test=None,
            all_y_trues_test=None,
        ):
            losses = [100000000.0]
            test_losses = [100000000.0]
            parameter_counts = []
            loss_de = 1000
            loss = 100000000.0
            n_sam = len(data[0, :, 0])
            epoch = 0
            while (loss_de >= pre_term and losses[-1] >= pre_ac) and epoch < max_epochs:
                all_y_pred = self.feedforward(data, j)
                activ = self.sinc(data, j, self.k)
                y_L_y_w1_1 = [1 * np.dot(activ[0, :, :], all_y_pred - all_y_trues) / n_sam]
                y_L_y_w1_2 = [1 * np.dot(activ[1, :, :], all_y_pred - all_y_trues) / n_sam]
                y_L_y_w1 = np.concatenate((y_L_y_w1_1, y_L_y_w1_2), 1).reshape(-1)
                self.w1 = self.w1 - learn_rate * y_L_y_w1
                parameter_counts.append(len(self.w1))
                loss_me = loss
                loss = mse_loss(all_y_trues, all_y_pred)
                losses.append(loss)
                if data_test is not None and all_y_trues_test is not None:
                    test_y_pred = self.feedforward(data_test, j)
                    test_loss = mse_loss(all_y_trues_test, test_y_pred)
                    test_losses.append(test_loss)
                epoch += 1
                loss_de = abs(loss_me - loss)
            return (self.w1, epoch, losses, test_losses, parameter_counts)


    @accelerator.numpy_class
    class ExpandedNetwork:


        def __init__(self, N_w1, w_need, k2, j, j_init):
            self.j_init = j_init
            self.w1 = np.array(w_need[:N_w1])
            self.w2_old = np.array(w_need[N_w1:])
            self.k2_old = k2
            self.k = []
            for num_ in range(j_init, j + 1):
                for i in range(2 * 2**num_ + 1):
                    for j_i in range(2 * 2**num_ + 1):
                        self.k.append([i, j_i])
            self.k = np.array(self.k)
            shape = self.k.shape
            self.k = self.k.reshape(shape[0], 1, shape[1])
            if len(self.k2_old) == 0:
                self.k2 = np.array([])
            else:
                self.k2 = self.k2_old

        def sinc(self, x, j, k):
            y = np.sum((2**j * x - k) ** 2, 2) ** 0.5 + np.pi / 1000000.0
            return np.array(
                [2**j * (np.sin(y) / y), 2**j * (-np.sin(2 * y) / y + np.sin(y) / y)]
            )

        def feedforward(self, x, j):
            M1_init = self.sinc(x, self.j_init, self.k[: (2 * 2**self.j_init + 1) ** 2, :, :])
            M1 = np.concatenate((M1_init[0, :, :], M1_init[1, :, :]), 0)
            for num_ in range(self.j_init + 1, j + 1):
                M1 = np.concatenate(
                    (
                        M1,
                        self.sinc(
                            x,
                            num_,
                            self.k[
                                (2 * 2 ** (num_ - 1) + 1) ** 2 : (2 * 2 ** (num_ - 1) + 1) ** 2
                                + (2 * 2**num_ + 1) ** 2,
                                :,
                                :,
                            ],
                        )[1, :, :],
                    ),
                    0,
                )
            M2 = self.sinc(x, j + 1, self.k2)
            return np.dot(M1.T, self.w1) + np.dot(M2[1, :, :].T, self.w2)

        def nearest_child_coordinates(self, x1, x2, j):
            ind = np.linspace(0, 4 * 2 ** (j - 1), int(4 * 2 ** (j - 1)) + 1)
            x1_w = np.ones((len(x1), 2))
            x2_w = np.ones((len(x1), 2))
            for i in range(len(x1)):
                x1_w[i, :2] = ind[np.where(abs(x1[i] * 2**j - ind) <= 1)][:2]
                x2_w[i, :2] = ind[np.where(abs(x2[i] * 2**j - ind) <= 1)][:2]
            return np.hstack((x1_w, x2_w))

        def train(
            self,
            data,
            all_y_trues,
            m_0,
            pre_ac,
            pre_term,
            epoch_last,
            learn_rate,
            data_test=None,
            all_y_trues_test=None,
        ):
            child_centers = np.array([[-100, -100]])
            k2_old_len = 0
            if len(self.k2_old) != 0:
                child_centers = np.vstack((child_centers, self.k2_old.reshape(-1, 2)))
                k2_old_len = len(self.k2_old)
            physical_parent_centers = [-200, -200]
            for i in range(2 * 2**j + 1):
                for i_0 in range(2 * 2**j + 1):
                    physical_parent_centers = np.vstack(
                        (physical_parent_centers, [i / 2**j, i_0 / 2**j])
                    )
            physical_parent_centers = np.vstack(
                (physical_parent_centers[1:, 0:2], physical_parent_centers[1:, 0:2])
            )
            m_0_len = m_0.shape[0]
            x_w = self.nearest_child_coordinates(
                physical_parent_centers[m_0, 0], physical_parent_centers[m_0, 1], j + 1
            )
            for i in range(m_0_len):
                for i_1 in [0, 1]:
                    for i_2 in [2, 3]:
                        isin = np.any(np.all(x_w[i, [i_1, i_2]] == child_centers, axis=1))
                        if isin == 0:
                            child_centers = np.vstack((child_centers, x_w[i, [i_1, i_2]]))
            k2_new = child_centers[1 + k2_old_len :, 0:2]
            shape_2 = k2_new.shape
            self.k2_new = k2_new.reshape(shape_2[0], 1, shape_2[1])
            self.w2_new = np.random.normal(0, 0, shape_2[0])
            if len(self.k2_old) == 0:
                self.k2 = self.k2_new
            else:
                self.k2 = np.vstack((self.k2_old, self.k2_new))
            self.w2 = np.vstack((self.w2_old.reshape(-1, 1), self.w2_new.reshape(-1, 1))).reshape(
                -1
            )
            parameter_counts = []
            loss_de = 1000
            loss = 100000000.0
            n_sam = len(data[0, :, 0])
            epoch = epoch_last
            mt1 = np.zeros(self.w1.shape[0])
            vt1 = np.zeros(self.w1.shape[0])
            mt2 = np.zeros(self.w2.shape[0])
            vt2 = np.zeros(self.w2.shape[0])
            while (losses[-1] >= pre_ac and loss_de >= pre_term) and epoch < max_epochs:
                all_y_pred = self.feedforward(data, j)
                activ1_init = self.sinc(
                    data, self.j_init, self.k[: (2 * 2**self.j_init + 1) ** 2, :, :]
                )
                y_L_y_w1_1 = [1 * np.dot(activ1_init[0, :, :], all_y_pred - all_y_trues) / n_sam]
                y_L_y_w1_2 = [1 * np.dot(activ1_init[1, :, :], all_y_pred - all_y_trues) / n_sam]
                y_L_y_w1 = np.concatenate((y_L_y_w1_1, y_L_y_w1_2), 1)
                for num_ in range(self.j_init + 1, j + 1):
                    activ1 = self.sinc(
                        data,
                        num_,
                        self.k[
                            (2 * 2 ** (num_ - 1) + 1) ** 2 : (2 * 2 ** (num_ - 1) + 1) ** 2
                            + (2 * 2**num_ + 1) ** 2,
                            :,
                            :,
                        ],
                    )
                    y_L_y_w1 = np.concatenate(
                        (y_L_y_w1, [1 * np.dot(activ1[1, :, :], all_y_pred - all_y_trues) / n_sam]),
                        1,
                    )
                y_L_y_w1 = y_L_y_w1.reshape(-1)
                activ2 = self.sinc(data, j + 1, self.k2)
                y_L_y_w2_2 = [1 * np.dot(activ2[1, :, :], all_y_pred - all_y_trues) / n_sam]
                y_L_y_w2 = np.array(y_L_y_w2_2).reshape(-1)
                mt1 = 0.9 * mt1 + (1 - 0.9) * y_L_y_w1
                vt1 = 0.999 * vt1 + (1 - 0.999) * y_L_y_w1**2
                self.w1 = self.w1 - learn_rate * mt1 / (np.sqrt(vt1) + 1e-07)
                mt2 = 0.9 * mt2 + (1 - 0.9) * y_L_y_w2
                vt2 = 0.999 * vt2 + (1 - 0.999) * y_L_y_w2**2
                self.w2 = self.w2 - learn_rate * mt2 / (np.sqrt(vt2) + 1e-07)
                parameter_counts.append(len(self.w1) + len(self.w2))
                loss_me = loss
                loss = mse_loss(all_y_trues, all_y_pred)
                losses.append(loss)
                if data_test is not None and all_y_trues_test is not None:
                    test_y_pred = self.feedforward(data_test, j)
                    test_loss = mse_loss(all_y_trues_test, test_y_pred)
                    test_losses.append(test_loss)
                epoch += 1
                loss_de = abs(loss_me - loss)
            return ([self.w1, self.w2], self.k2, epoch, losses, test_losses, parameter_counts)

    mu = settings.get("mu", 1 / 5)
    tran_r = 2
    dimension = 2
    j_init = settings.get("j_init", 2)
    j = j_init
    N_w1 = int((tran_r * 2**j + 1) ** dimension + (tran_r * 2**j + 1) ** dimension)
    pre_ac = settings.get("pre_ac", 0.0025)
    pre_term = settings.get("pre_term", 4e-05)
    learn_rate = settings.get("learn_rate", 0.0001)
    start_time = time.time()
    network = InitialNetwork(N_w1)
    (w_need, epoch, losses, test_losses, parameter_counts) = network.train(
        data_train, all_y_trues_train, j, pre_ac, pre_term, learn_rate, data_test, all_y_trues_test
    )
    while losses[-1] >= pre_ac and epoch < max_epochs:
        k2 = np.array([])
        m_0_ind = 0
        while (losses[-1] >= pre_ac and m_0_ind <= int(1 / mu) - 1) and epoch < max_epochs:
            m_0 = energy_separate(w_need, mu, j, tran_r, dimension)[m_0_ind]
            network_adding = ExpandedNetwork(N_w1, w_need, k2, j, j_init)
            (
                w_need,
                k2,
                epoch,
                losses,
                test_losses,
                parameter_counts_adding,
            ) = network_adding.train(
                data_train,
                all_y_trues_train,
                m_0,
                pre_ac,
                pre_term,
                epoch,
                learn_rate,
                data_test,
                all_y_trues_test,
            )
            w_need = np.vstack((w_need[0].reshape(-1, 1), w_need[1].reshape(-1, 1))).reshape(-1)
            parameter_counts.extend(parameter_counts_adding)
            m_0_ind += 1
        j = j + 1
        N_w1 = N_w1 + int((tran_r * 2**j + 1) ** dimension)
    end_time = time.time()
    test_losses = test_losses[1:]
    losses = losses[1:]
    time_long = end_time - start_time
    return locals()
