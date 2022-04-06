import numpy as np

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.all_clients = []
        self.updates = []
        self.L = {}
        self.N = {}
        self.A = {}
        self.p = {}
        self.T = 0
        self.gamma = 0.9

    def initialize_dicts(self, possible_clients):
        for c in possible_clients:
            self.L[c] = 0
            self.N[c] = 0
            self.A[c] = 0

    def policy0(self, possible_clients, num_clients, my_round):
        np.random.seed(my_round)
        selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return selected_clients.copy()

    def policy1(self, possible_clients, num_clients, my_round):
        if my_round ==0:
            _,_, num_samples = self.get_clients_info(possible_clients)
            self.p = num_samples.copy()
            p_sum = sum(self.p.values())
            for c in self.p.keys():
                self.p[c] = self.p[c]/p_sum

            return possible_clients.copy()

        self.T = 1 + self.gamma*self.T
        for c in self.all_clients:
            self.L[c] = c.loss_history[-1] + self.gamma*self.L[c]
            self.N[c] = int(c in self.selected_clients) + self.gamma*self.N[c]
            self.A[c] = self.p[c.id]*(self.L[c]/self.N[c] + np.sqrt(2*np.log(self.T)/self.N[c]))

        sorted_A = sorted(self.A.items(), key = lambda kv: kv[1], reverse = True)
        selected_clients = []
        for i in range(num_clients):
            selected_clients.append(sorted_A[i][0])

        return selected_clients.copy()



    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        self.all_clients = possible_clients.copy()
        num_clients = min(num_clients, len(possible_clients))
        #self.selected_clients = self.policy0(possible_clients, num_clients, my_round)
        self.selected_clients = self.policy1(possible_clients, num_clients, my_round)
        for c in possible_clients:
            c.loss_history.append(0)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            c.model.set_params(self.model)
            comp, num_samples, update, loss = c.train(num_epochs, batch_size, minibatch)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))
            c.loss_history[-1] = loss

        print("No. of clients used: ", len(self.updates))

        return sys_metrics

    def update_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()