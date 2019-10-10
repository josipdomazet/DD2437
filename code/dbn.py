from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():

    '''
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),

            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 200

        self.n_gibbs_wakesleep = 15

        self.print_period = 2000

        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        vis = true_img # visible layer gets the image data

        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels

        input2 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[0]
        input3 = self.rbm_stack["hid--pen"].get_h_given_v_dir(input2)[0]

        input3 = np.hstack((input3, true_lbl))
        #  binary sample representations for 500 units
        for _ in range(self.n_gibbs_recog):
            hidden3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(input3)[0]  # softmax
            input3 = self.rbm_stack["pen+lbl--top"].get_v_given_h(hidden3)[1]  # binary sample

        predicted_lbl2 = input3[:, -lbl.shape[1]:]
        print("accuracy = %.2f%%" % (100. * np.mean(np.argmax(predicted_lbl2, axis=1) == np.argmax(true_lbl, axis=1))))

        return

    def generate(self,true_lbl,name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]

        records = []
        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])
        lbl = true_lbl
        visible_top_rbm = np.random.rand(n_sample, self.sizes["pen"])
        for _ in range(self.n_gibbs_gener):
            hidden_top_rbm = self.rbm_stack["pen+lbl--top"].get_h_given_v(np.hstack((visible_top_rbm, lbl)))
            #  hid3 - always clamp true labels because we're forsing network to reproduce specific images
            #  HIDDEN UNITS have to binary
            visible_top_rbm = self.rbm_stack["pen+lbl--top"].get_v_given_h(hidden_top_rbm[1])[1]
            visible_top_rbm = visible_top_rbm[:, :-true_lbl.shape[1]]
            #  remove the labels because we're fixing true labels

        #  we freeze the probabilities so we can sample multiple times and get multiple pictures !
        for _ in range(self.n_gibbs_gener):
            sampled_hidden_top_rbm = sample_binary(hidden_top_rbm[0])
            visible_top_rbm = self.rbm_stack["pen+lbl--top"].get_v_given_h(sampled_hidden_top_rbm)[1]
            visible_top_rbm = visible_top_rbm[:, :-true_lbl.shape[1]]  # remove labels so it can be propagated to the bottom RBM
            visible_middle_rbm = self.rbm_stack["hid--pen"].get_v_given_h_dir(visible_top_rbm)[1]
            visible_bottom_rbm = self.rbm_stack["vis--hid"].get_v_given_h_dir(visible_middle_rbm)[0]
            records.append( [ ax.imshow(visible_bottom_rbm.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )

        anim = stitch_video(fig,records).save("videos/%s.generate%d.mp4"%(name,np.argmax(true_lbl)))

        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")
            # top layer has bidirectional weights!

        except IOError :

            print ("training vis--hid")
            """
            CD-1 training for vis--hid
            """
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            """
            CD-1 training for hid--pen
            """

            input_rbm2 = self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)
            self.rbm_stack["hid--pen"].cd1(input_rbm2[1], n_iterations)
            # hidden units of RBM1 are visible units of RBM2, that's why input_rbm2[1] is used (sampled values)
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")

            print ("training pen+lbl--top")
            """
            CD-1 training for pen+lbl--top
            """

            input_rbm3 = self.rbm_stack["hid--pen"].get_h_given_v(input_rbm2[1])
            input_rbm3 = np.hstack((input_rbm3[1], lbl_trainset))  # need to add labels

            self.rbm_stack["pen+lbl--top"].cd1(input_rbm3, n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")

            # decoupling weights for first two RBMs
            # top layer RBM weights are not decoupled!
            self.rbm_stack["vis--hid"].untwine_weights()
            self.rbm_stack["hid--pen"].untwine_weights()
        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("\ntraining wake-sleep..")

        try :

            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")

        except IOError :

            self.n_samples = vis_trainset.shape[0]
            batches_in_epoch = int(vis_trainset.shape[0] / self.batch_size)
            batch_index = 0
            for it in range(n_iterations):

                """
                wake-phase : drive the network bottom-to-top using visible and label data
                """

                ##### Bottom RMB #####
                input_rbm1 = vis_trainset[self.batch_size*batch_index:self.batch_size*(batch_index+1), :]
                # use recognition weights (visible ---> hidden) to pick hidden nodes
                hidden_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(input_rbm1)
                # reconstruct original visible nods using generative weights (hidden ---> visible)
                reconstructed_1 = self.rbm_stack["vis--hid"].get_v_given_h_dir(hidden_1[1])
                #  generative weights will be adjusted later

                ##########################################################

                ##### Middle RBM #####
                input_rbm2 = hidden_1[1]  # binary
                hidden_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(input_rbm2)
                reconstructed_2 = self.rbm_stack["hid--pen"].get_v_given_h_dir(hidden_2[1])

                ##########################################################

                ##### Top RMB (labeled) #####
                label_batch = lbl_trainset[self.batch_size*batch_index:self.batch_size*(batch_index+1), :]
                #  add the labels!
                input_rbm3 = np.hstack((hidden_2[1], label_batch))
                initial_visible = np.copy(input_rbm3)
                initial_hidden = self.rbm_stack["pen+lbl--top"].get_h_given_v(input_rbm3)
                n_gibbs_visible = np.copy(input_rbm3)
                n_gibbs_hidden = None
                for _ in range(self.n_gibbs_wakesleep):
                    n_gibbs_hidden = self.rbm_stack["pen+lbl--top"].get_h_given_v(n_gibbs_visible)
                    n_gibbs_visible = self.rbm_stack["pen+lbl--top"].get_v_given_h(n_gibbs_hidden[1])[1]
                hidden_2_orig = n_gibbs_visible[:, :-lbl_trainset.shape[1]]
                visible_2_down = self.rbm_stack["hid--pen"].get_v_given_h_dir(hidden_2_orig)
                hidden_2_down_pred = self.rbm_stack["hid--pen"].get_h_given_v_dir(visible_2_down[1])

                hidden_1_orig = visible_2_down[1]
                visible_1_down = self.rbm_stack["vis--hid"].get_v_given_h_dir(hidden_1_orig)
                hidden_1_down_pred = self.rbm_stack["vis--hid"].get_h_given_v_dir(visible_1_down[0])

                self.rbm_stack["vis--hid"].update_generate_params(hidden_1[1], input_rbm1, reconstructed_1[0])
                self.rbm_stack["hid--pen"].update_generate_params(hidden_2[1], input_rbm2, reconstructed_2[0])
                self.rbm_stack["pen+lbl--top"].update_params
                (initial_visible, initial_hidden[1], n_gibbs_visible, n_gibbs_hidden[1])
                self.rbm_stack["hid--pen"].update_recognize_params(visible_2_down[1], hidden_2_orig, hidden_2_down_pred[0])
                self.rbm_stack["vis--hid"].update_recognize_params(visible_1_down[1], hidden_1_orig,
                                                                   hidden_1_down_pred[0])

                batch_index = (batch_index + 1) % batches_in_epoch

                if it % self.print_period == 0 : print ("iteration=%7d"%it)

            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")

        return


    def loadfromfile_rbm(self,loc,name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return

    def savetofile_rbm(self,loc,name):
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self,loc,name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return

    def savetofile_dbn(self,loc,name):

        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
