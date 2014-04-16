import sys
import mapper
import time

import pybrain
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer
from pybrain.tools.customxml import networkwriter
#from pybrain.tools.xml import networkwriter

class PyImpNetwork:

    def __init__(self):
        #flags for program learning states
        self.learning = 0
        self.compute = 0
        self.recurrent_flag = False; # default case is a nonrecurrent feedforward network

        #number of mapper inputs and outputs
        self.num_inputs = 0
        self.num_outputs = 0
        self.num_hidden = 0

        #For the Mapper Signals
        self.l_inputs = {}
        self.l_outputs = {}

        #For the Artificial Neural Network
        self.data_input = {}
        self.data_output = {}
        self.data_middle = {}

        # store a list of the signals obtained from the device
        self.input_names = []
        self.output_names = []

        #temporary array for storying single snapshots
        self.temp_ds = {}
        self.snapshot_count = 0

        self.learnMapperDevice = mapper.device("Implicit_LearnMapper",9000)
        #self.learnMapperDevice = mapper.device("Implicit_LearnMapper")

    # mapper signal handler (updates self.data_input[sig_indx]=new_float_value)
    def h(self,sig, f):

        if f == None:
            return

        try:
            if '/in' in sig.name:
                s_indx = str.split(sig.name,"/in")
                self.data_input[int(s_indx[1])] = float(f)

                if sig.name not in self.input_names:
                    self.input_names.append(sig.name)   

            elif '/out' in sig.name:
                    s_indx = str.split(sig.name,"/out")
                    self.data_output[int(s_indx[1])] = float(f)
                    #print "Output Value from data_output", self.data_output[int(s_indx[1])]
            
            #print "Output Value from data_output", self.data_output.values()
            #print self.input_names 
        except Exception, e:
           print "Exception, Handler not working:", e

    def createANN(self,n_inputs,n_hidden,n_outputs):
        #create ANN
        self.net = buildNetwork(n_inputs,n_hidden,n_outputs,bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, recurrent=self.recurrent_flag)
        
        #create ANN Dataset
        self.ds = SupervisedDataSet(n_inputs,n_outputs)

    def createMapperInputs(self,n_inputs):
        #create mapper signals (inputs)
        for l_num in range(n_inputs):
            print ("creating input", "/in"+str(l_num))
            self.l_inputs[l_num] = self.learnMapperDevice.add_input("/in"+str(l_num),1,'f', "Hz", 0.0, 1.0,self.h)

        # Set initial Data Input values for Network to 0
        for s_index in range(n_inputs):
            self.data_input[s_index] = 0.0

    def createMapperOutputs(self,n_outputs):
        #create mapper signals (n_outputs)
        for l_num in range(n_outputs):
            self.l_outputs[l_num] = self.learnMapperDevice.add_output("/out%d"%l_num,1,'f', None, 0.0, 1.0)
            #self.l_outputs[l_num].set_query_callback(self.h)
            print ("creating output","/out"+str(l_num))
        
        # Set initial Data Output values for Network to 0
        for s_index in range (n_outputs):
            self.data_output[s_index] = 0.0

    def setNumInputs(self,n_inputs):
        self.num_inputs = n_inputs

    def setNumeOutputs(self,n_outputs):
        self.num_outputs = n_outputs

    def setNumHiddenNodes(self,n_hidden):
        self.num_hidden = n_hidden

    def setReccurentFlag(self,flag):
        if (flag == "R"):
            self.recurrent_flag=True
        elif (flag == "F"):
            self.recurrent_flag=False
  
    def load_dataset(self,open_filename):
        self.ds = SupervisedDataSet.loadFromFile(open_filename)
        #print self.ds

    def save_dataset(self,filename):

        if str(filename[0]) != '': 
            csv_file = open(filename[0]+".csv", "w")
            csv_file.write("[inputs][outputs]\r\n")
        
        for inpt, tgt in self.ds:
                new_str=str("{" + repr(inpt) + "," + repr(tgt) + "}")
                new_str=new_str.strip('\n')
                new_str=new_str.strip('\r')
                new_str=new_str+"\r"
                csv_file.write(new_str)

        if len(new_str)>1: 
            csv_file.close()

    def save_net(self,save_filename):
        networkwriter.NetworkWriter.writeToFile(net,save_filename)

    def load_net(self,open_filename):
        from pybrain.tools.customxml import networkreader
        self.net = networkreader.NetworkReader.readFrom(open_filename)

    def clear_dataset(self):

        if self.temp_ds != 0:
            self.temp_ds.clear()
            self.snapshot_count = 0

        if self.ds != 0:
            self.ds.clear()

    def clear_network(self):
        #resets the module buffers but doesn't reinitialise the connection weights
        #TODO: reinitialise network here or make a new option for it.
        self.net.reset()

    def learn_callback(self):

        # Save data to a temporary database in case they need to be edited before adding to the Supervised Dataset
        self.snapshot_count = self.snapshot_count+1
        self.temp_ds[self.snapshot_count] = {}
        self.temp_ds[self.snapshot_count]["input"] = tuple(self.data_input.values())
        self.temp_ds[self.snapshot_count]["output"] = tuple(self.data_output.values())
        self.update_ds()

        print "Values before going to temp_ds", self.data_input.values(), "   ", self.data_output.values()
        print self.snapshot_count, "(Input, Output)", self.temp_ds[self.snapshot_count]


    def remove_tempds(self,objectNum):

        if objectNum in self.temp_ds.iterkeys():
            print "Found DS to delete", objectNum
            del self.temp_ds[objectNum]

            if self.snapshot_count > (-1):
                self.snapshot_count = self.snapshot_count - 1

        else: 
            print "Error, This database entry does not exist"


    def compute_callback(self):

        activated_out = self.net.activate(tuple(self.data_input.values()))
        for out_index in range(self.num_outputs):
            self.data_output[out_index] = activated_out[out_index]
            self.l_outputs[out_index].update(self.data_output[out_index])

    def train_callback(self):
        self.trainer = BackpropTrainer(self.net, learningrate=0.01, lrdecay=1, momentum=0.0, verbose=True)
        
        print 'MSE before', self.trainer.testOnData(self.ds, verbose=True)
        epoch_count = 0
        while epoch_count < 1000:
            epoch_count += 10
            self.trainer.trainUntilConvergence(dataset=self.ds, maxEpochs = 10)
            networkwriter.NetworkWriter.writeToFile(self.net,'autosave.network')
        
        print 'MSE after', self.trainer.testOnData(self.ds, verbose=True)
        print ("\n")
        print 'Total epochs:', self.trainer.totalepochs

    def update(self):
        
        # Compute the output from the input values
        if self.compute == 1: 
            self.compute_callback()

        else: 
            # Update the Output: Get Value from the output by sending a query request
            for index in range(self.num_outputs):
                pass
                #self.l_outputs[index].query_remote()

    def update_ds(self):
        if self.ds != 0:
            self.ds.clear()

        for key in sorted(self.temp_ds.iterkeys()): 
            self.ds.addSample(self.temp_ds[key]["input"],self.temp_ds[key]["output"])
