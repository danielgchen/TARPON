from collections import deque
import neuron as neuron
import numpy as np
import utils as utils

# retrieve constants from utils
function_to_activation = utils.retrieve_f2a()
function_to_derivative = utils.retrieve_f2d()

class NeuronNetwork:
    """
    defines a neural network to connect nodes in an acyclic directed graph
    """
    def __init__(self, n_inputs, n_outputs, hidden_layers):
        """
        neural network defined by inputs, outputs, and at least one hidden layer
        """
        # define the inputs, outputs, and hidden layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        # compute the total number of nodes
        self.n_nodes = sum(hidden_layers) + n_outputs
        # calculate the output nids
        self.output_nids = [self.n_nodes - (self.n_outputs - idx) for idx in range(self.n_outputs)]
        # create connection matrix from rows (inputs) to outputs (columns)
        self.adj = np.zeros(shape=(self.n_nodes, self.n_nodes), dtype=int)
        # create map of nid to node
        # > create initial hidden layer
        self.nodes = {idx: neuron.Neuron(nid=idx, n_inputs=n_inputs) for idx in range(hidden_layers[0])}
        # > note the size of the first hidden as these will be the true inputs
        self.first_nids = list(range(hidden_layers[0]))
        # > define method to add hidden layer
        def _add_hidden_layer(hidden_layers, idx1, n_nodes):
            """
            method to add a hidden layer
            retrieves idx1 from hidden_layers and add n_nodes
            """
            # retrieve the current number of nodes
            # , for nid purposes
            size = len(self.nodes)
            # retrieve the number of entries in the previous hidden layer
            # , to determine the number of input nodes, initially
            n_inputs = hidden_layers[idx1]
            # create the nodes for this layer
            self.nodes.update({size+idx: neuron.Neuron(nid=size+idx, n_inputs=n_inputs) for idx in range(n_nodes)})
            # map nodes via the adjacency matrix
            self.adj[size-n_inputs:size, size:size+n_nodes] = 1
        # > create rest of hidden layers if they exist
        for idx1, n_nodes in enumerate(hidden_layers[1:]):
            _add_hidden_layer(hidden_layers, idx1, n_nodes)
        # create output layer
        _add_hidden_layer(hidden_layers, -1, n_outputs)
        # create a list of nodes that can be used post removal
        self.free_nids = []
        # the next free nid after initialization will be the length
        self.next_nid = self.n_nodes
        
    def forward(self, X):
        # create the queue to search through
        # , use the out nids for each of the nids of the first hidden layer
        queue = deque()
        # work through the first hidden layer
        for nid in self.first_nids:
            # process the node
            self.nodes[nid].process(X)
            # add it's out nids if they aren't already in the queue
            out_nids = np.nonzero(self.adj[nid, :])[0]
            queue.extend([out_nid for out_nid in out_nids if out_nid not in queue])
        # create tracking variables
        processed = np.array(self.first_nids)

        # keep processing until every node has been touched
        while queue:
            # remove the first node in the queue
            nid = queue.popleft()
            # if we already processed this node move on
            if nid in processed:
                continue
            # if we cannot process this node move on
            in_nids = np.nonzero(self.adj[:, nid])[0]
            if not np.isin(in_nids, processed).all():
                queue.append(nid)
                continue
            # process if satisfy both conditions
            # > add to array of things processed
            processed = np.append(processed, nid)
            # > process the node
            ins = np.vstack([self.nodes[in_nid].activate() for in_nid in in_nids]).T
            self.nodes[nid].process(ins)
            # > add outs to queue
            out_nids = np.nonzero(self.adj[nid, :])[0]
            queue.extend([out_nid for out_nid in out_nids if out_nid not in queue])
        # retrieve outputs
        outputs = np.vstack([self.nodes[output_nid].activate() for output_nid in self.output_nids]).T
        return outputs
    
    def backward(self, X, y, y_pred):
        # calculate the derivative of the loss function with respect to a
        # we will need to keep track of these values
        # , this is to map the cost with regards to a per node
        cost_wrt_a_map = {nid: 2 * (y_pred[:, idx] - y[:, idx]) for idx, nid in enumerate(self.output_nids)}
        # retrieve the total number of samples
        n_samples = X.shape[0]
        # create the queue to search through
        # , the input nids for the outputs will be specified here
        queue = deque()
        
        # work through the output layer
        for nid in self.output_nids:
            # retrieve the in nids
            # , to calculate the a values
            in_nids = np.nonzero(self.adj[:, nid])[0]
            ins = np.vstack([self.nodes[in_nid].activate() for in_nid in in_nids])
            # add the in nids to the queue if they are not there
            queue.extend([in_nid for in_nid in in_nids if in_nid not in queue])
            # solve for the gradient
            # , this will be z/w * a/z * c/a
            # , z/w is ins, a/z is a scalar, c/a = will be per sample like z/w
            coefs = self.nodes[nid].derivative() * cost_wrt_a_map[nid]
            gradient = np.mean(ins * coefs, axis=1)
            # multiply by learning rate
            gradient *= self.learning_rate
            # adjust the weight
            self.nodes[nid].weights -= gradient

        # create tracking variables
        processed = np.array(self.output_nids)

        # keep processing until every node has been touched
        while queue:
            # remove the first node in the queue
            nid = queue.popleft()
            # if nid is one of the input processing nids move on
            if nid in self.first_nids:
                continue
            # if we already processed this node move on
            if nid in processed:
                continue
            # if we cannot process this node move on
            out_nids = np.nonzero(self.adj[nid, :])[0]
            if not np.isin(out_nids, processed).all():
                queue.append(nid)
                continue
            # process if satisfy both conditions
            # > add to array of things processed
            processed = np.append(processed, nid)
            # > process the node
            outs = np.vstack([self.nodes[out_nid].activate() for out_nid in out_nids]).T       
            # , to calculate the a values
            in_nids = np.nonzero(self.adj[:, nid])[0]
            ins = np.vstack([self.nodes[in_nid].activate() for in_nid in in_nids])
            # > add the in nids to the queue if they are not there
            queue.extend([in_nid for in_nid in in_nids if in_nid not in queue])
            # > solve for the gradient
            # , this will be z/w * a/z * c/a
            # , z/w is ins, a/z is a scalar, c/a = will be per sample like z/w
            # , c/a is special because it is not directly linked to the output
            # , so instead it is based on the chain of outs which is stored in the c2a map
            cost_wrt_as = []
            for out_nid in out_nids:
                in_idxs_for_out = np.nonzero(self.adj[:, out_nid])[0]
                in_idx_for_out = np.where(in_idxs_for_out == nid)[0][0]
                cost_wrt_as.append(self.nodes[out_nid].weights[in_idx_for_out] * self.nodes[out_nid].derivative() * cost_wrt_a_map[out_nid])
            cost_wrt_a = np.vstack(cost_wrt_as).sum(axis=0)
            # >> we will save this c2a value for upstream nodes
            cost_wrt_a_map[nid] = cost_wrt_a
            coefs = self.nodes[nid].derivative() * cost_wrt_a
            gradient = np.mean(ins * coefs, axis=1)
            # > multiply by learning rate
            gradient *= self.learning_rate
            # > adjust the weight
            self.nodes[nid].weights -= gradient
            
        # process the input processing nids
        for nid in self.first_nids:
            # if we cannot process this node move on
            out_nids = np.nonzero(self.adj[nid, :])[0]
            if not np.isin(out_nids, processed).all():
                queue.append(nid)
                continue
            # process if satisfy both conditions
            # > add to array of things processed
            processed = np.append(processed, nid)
            # > process the node
            outs = np.vstack([self.nodes[out_nid].activate() for out_nid in out_nids]).T            
            # > solve for the gradient
            # , this will be z/w * a/z * c/a
            # , z/w is ins, a/z is a scalar, c/a = will be per sample like z/w
            # , c/a is special because it is not directly linked to the output
            # , so instead it is based on the chain of outs which is stored in the c2a map
            cost_wrt_as = []
            for out_nid in out_nids:
                in_idxs_for_out = np.nonzero(self.adj[:, out_nid])[0]
                in_idx_for_out = np.where(in_idxs_for_out == nid)[0][0]
                cost_wrt_as.append(self.nodes[out_nid].weights[in_idx_for_out] * self.nodes[out_nid].derivative() * cost_wrt_a_map[out_nid])
            cost_wrt_a = np.vstack(cost_wrt_as).sum(axis=0)
            # >> we will save this c2a value for upstream nodes
            cost_wrt_a_map[nid] = cost_wrt_a
            coefs = self.nodes[nid].derivative() * cost_wrt_a
            gradient = np.mean(X.T * coefs, axis=1)
            # > multiply by learning rate
            gradient *= self.learning_rate
            # > adjust the weight
            self.nodes[nid].weights -= gradient
    
    def train(self, X, y, epochs, initial_learning_rate=0.1, tolerance=0.0001, save_params=False):
        # set tracking variable for loss
        loss_curve = []
        # save initial parameters
        if save_params:
            parameters = [[w for nid in self.nodes for w in self.nodes[nid].weights]]
        # work through epochs or until convergence
        for epoch in range(epochs):
            # predict the output based on X
            y_pred = self.forward(X)
            # save the current loss
            self.current_loss = utils.calc_loss_sse(y_pred, y)
            loss_curve.append(self.current_loss)
            # if it's the first epoch force it to run at least one round
            if epoch == 0:
                original_loss = self.current_loss + tolerance + 1
            # check for convergence based on change in loss relative to tolerance
            if abs(original_loss - self.current_loss) < tolerance:
                print('model converged')
                if save_params:
                    return loss_curve, parameters
                return loss_curve
            # setup the learning rate based on the loss value
            self.learning_rate = initial_learning_rate * self.current_loss
            # utilize the truth to learn from the output
            self.backward(X, y, y_pred)
            # save the parameters if needed
            if save_params:
                parameters.append([w for nid in self.nodes for w in self.nodes[nid].weights])
            # save the current loss as the previous to be checked against next round
            original_loss = self.current_loss
        # model did not converge but max epochs reached
        print('model did not converge')
        if save_params:
            return loss_curve, parameters
        return loss_curve
    
    def predict(self, X):
        return self.forward(X)
    
    # non-classical functions
    def add_weight(self, nid1, nid2, weight=None):
        # skip if there is a connection to begin with
        if self.adj[nid1, nid2] == 1:
            print('there is a connection between', nid1, 'and', nid2)
            return
        # if there is not a connection first retrieve the in nids
        in_idxs_for_out = np.nonzero(self.adj[:, nid2])[0]
        # then map out which index the new nid would be inserted at
        in_idx_for_out = np.argsort(np.append(in_idxs_for_out, nid1))[-1]
        # add the connection
        self.adj[nid1, nid2] = 1
        # use a random weight if unspecified
        weight = np.random.randn(1)[0] if weight is None else weight
        # then add the weight to the receiving node
        self.nodes[nid2].weights = np.insert(self.nodes[nid2].weights, in_idx_for_out, weight)
        
    def remove_weight(self, nid1, nid2):
        # skip if there is not a connection to begin with
        if self.adj[nid1, nid2] == 0:
            print('there is no connection between', nid1, 'and', nid2)
            return
        # if there is a connection first retrieve the in nids
        in_idxs_for_out = np.nonzero(self.adj[:, nid2])[0]
        in_idx_for_out = np.where(in_idxs_for_out == nid1)[0][0]
        # remove the connection
        self.adj[nid1, nid2] = 0
        # then remove the weight from the receiving node
        self.nodes[nid2].weights = np.delete(self.nodes[nid2].weights, in_idx_for_out)
        
    def add_node(self):
        # reuse a free nid if possible
        if self.free_nids:
            # remove the free nid
            nid = self.free_nids.pop()
        # if not able to then increase the size of everything
        else:
            # move forward in the next nid counting
            nid = self.n_nodes
            self.next_nid += 1
            # adjust the adjacency matrix
            self.adj = np.hstack([self.adj, np.zeros((self.adj.shape[0], 1), dtype=int)])
            self.adj = np.vstack([self.adj, np.zeros((1, self.adj.shape[1]), dtype=int)])
        # we add it from the node list
        self.nodes[nid] = neuron.Neuron(nid=nid, n_inputs=0)
        self.n_nodes += 1
        return nid
        
    def remove_node(self, nid):
        # this should only occur if it is not attached to anything
        if (self.adj[nid].sum() > 0) or (self.adj[:, nid].sum() > 0):
            print(nid, 'is still connected to other nodes')
            return
        # if it is unattached then we can safely remove it
        # we "remove" it from the adjacency matrix
        self.free_nids.append(nid)
        # we remove it from the node list
        del self.nodes[nid]
        self.n_nodes -= 1