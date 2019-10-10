from util import *
import numpy as np

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size

        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = 1000

        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 500, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }

        return


    def cd1(self,visible_trainset, n_iterations=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")
        batches_in_epoch = int(visible_trainset.shape[0]/self.batch_size)
        batch_index = 0
        f = open('rbm_results_' + str(self.ndim_hidden) + '.csv', 'w+')
        f.write("iteration,result\n")
        for it in range(n_iterations):
            mini_batch = visible_trainset[self.batch_size*batch_index:self.batch_size*(batch_index+1), :]
            batch_index = (batch_index + 1) % batches_in_epoch
            # positive phase
            hidden_units = self.get_h_given_v(mini_batch)
            # negative phase
            reconstr_mini_batch = self.get_v_given_h(hidden_units[1])
            hidden_of_reconstructed_input = self.get_h_given_v(reconstr_mini_batch[0])  # [0] is probability
            # updating parameters
            self.update_params(mini_batch, hidden_units[1], reconstr_mini_batch[0], hidden_of_reconstructed_input[0])
            # commented part is for visualising receptive fields
            # visualize once in a while when visible layer is input images
#           if it % self.rf["period"] == 0 and self.is_bottom:
#                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # print progress
            if it % self.print_period == 0 :
                rez = np.sum((reconstr_mini_batch[0] - mini_batch)**2) / (self.ndim_visible * self.batch_size)
                f.write(str(it) + ',' + str(round(rez, 4)) + '\n')
                print ("iteration=%7d recon_loss=%4.4f"%(it, rez))

        f.close()
        return

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        self.delta_weight_vh = np.dot(v_0.T, h_0) - np.dot(v_k.T, h_k)
        self.delta_bias_h = np.mean(h_0 - h_k, axis=0)
        self.delta_bias_v = np.mean(v_0 - v_k, axis=0)

        self.bias_v += self.delta_bias_v * self.learning_rate
        self.bias_h += self.delta_bias_h * self.learning_rate

        self.weight_vh += self.delta_weight_vh * self.learning_rate / self.batch_size
        return

    def get_h_given_v(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None
        probs_h_given_v = np.matmul(visible_minibatch, self.weight_vh)
        probs_h_given_v = sigmoid(probs_h_given_v + self.bias_h)
        activations = sample_binary(probs_h_given_v)
        return probs_h_given_v, activations


    def get_v_given_h(self,hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            support = np.matmul(hidden_minibatch, self.weight_vh.T)
            support_labels = softmax(support[:, -self.n_labels:] + self.bias_v[-self.n_labels:])
            support_rest = sigmoid(support[:, :-self.n_labels] + self.bias_v[:-self.n_labels])
            labels_activation = sample_categorical(support_labels) # CAREFUL!!!
            rest_activation = sample_binary(support_rest)
            activations = np.hstack((rest_activation, labels_activation))
            probabilities = np.hstack((support_rest, support_labels))
            return probabilities, activations

        else:
            probs_v_given_h = np.matmul(hidden_minibatch, self.weight_vh.T)  # the transposing of weights!
            probs_v_given_h = sigmoid(probs_v_given_h + self.bias_v)
            activations = sample_binary(probs_v_given_h)
            return probs_v_given_h, activations


    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """


    def untwine_weights(self):

        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None
        probs_h_given_v = np.matmul(visible_minibatch, self.weight_v_to_h)
        probs_h_given_v = sigmoid(probs_h_given_v + self.bias_h)
        activations = sample_binary(probs_h_given_v)
        return probs_h_given_v, activations

    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            pass

        else:
            probs_v_given_h = np.matmul(hidden_minibatch, self.weight_h_to_v)
            probs_v_given_h = sigmoid(probs_v_given_h + self.bias_v)
            activations = sample_binary(probs_v_given_h)
            return probs_v_given_h, activations


    def update_generate_params(self,inps,trgs,preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        self.delta_weight_h_to_v = np.dot(inps.T, (trgs - preds))

        self.delta_bias_v = np.mean(trgs - preds, axis=0)

        self.bias_v += self.delta_bias_v * self.learning_rate
        # need to include division by batch size since updating is based on average
        self.weight_h_to_v += self.delta_weight_h_to_v * self.learning_rate / self.batch_size

        return

    def update_recognize_params(self,inps,trgs,preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        self.delta_weight_v_to_h = np.dot(inps.T, (trgs - preds))
        self.delta_bias_h = np.mean(trgs - preds, axis=0)
        # need to include division by batch size since updating is based on average
        self.bias_h += self.delta_bias_h * self.learning_rate
        self.weight_v_to_h += self.delta_weight_v_to_h * self.learning_rate / self.batch_size

        return
