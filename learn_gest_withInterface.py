import sys
import mapper
import time

import pybrain
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer
from pybrain.tools.xml import networkwriter

import PySide
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtWebKit import *
from PySide.QtUiTools import *

class PyImpNetwork():

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

        self.learnMapperDevice = mapper.device("Implicit_LearnMapper",9002)

    # mapper signal handler (updates self.data_input[sig_indx]=new_float_value)
    def h(self,sig, f):
        try:
            print sig.name
            if '/in' in sig.name:
                s_indx=str.split(sig.name,"/in")
                self.data_input[int(s_indx[1])]=float(f)

            elif '/out' in sig.name:
                if (learning==1):
                    s_indx=str.split(sig.name,"/out")
                    self.data_output[int(s_indx[1])]=float(f)
        except:
            print "Exception, Handler not working"

    def createANN(self,n_inputs,n_hidden,n_outputs):
        #create ANN
        self.net = buildNetwork(n_inputs,n_hidden,n_outputs,bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, recurrent=self.recurrent_flag)
        
        #create ANN Dataset
        self.ds = SupervisedDataSet(n_inputs,n_outputs)

    def createMapperInputs(self,n_inputs):
        #create mapper signals (inputs)
        for l_num in range(n_inputs):
            self.l_inputs[l_num] = self.learnMapperDevice.add_input("/in%d"%l_num, 1, 'f',None,0,1.0, self.h)
            self.learnMapperDevice.poll(0)
            print ("creating input", "/in"+str(l_num))

        # Set initial Data Input values for Network to 0
        for s_index in range(n_inputs):
            self.data_input[s_index] = 0.0

    def createMapperOutputs(self,n_outputs):
        #create mapper signals (n_outputs)
        for l_num in range(n_outputs):
            self.l_outputs[l_num] = self.learnMapperDevice.add_output("/out"+str(l_num), 1, 'f',None,0.0,1.0)
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
        print self.ds

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
        if self.ds != 0:
            self.ds.clear()

    def clear_network(self):
        #resets the module buffers but doesn't reinitialise the connection weights
        #TODO: reinitialise network here or make a new option for it.
        self.net.reset()

    def learn_callback(self):

        if self.learning == 0:
            print ("learning is", self.learning)
            self.learning = 1

        elif self.learning == 1:
            print ("learning is", self.learning)
            self.learning = 0

    def compute_callback(self):

        if self.compute==1:
            self.compute =0
            print ("Compute network output is now OFF!")
        elif self.compute ==0:
            self.compute =1
            print ("Compute network output is now ON!")

    def train_callback(self):
        self.trainer = BackpropTrainer(self.net, learningrate=0.01, lrdecay=1, momentum=0.0, verbose=True)
        
        print 'MSE before', self.trainer.testOnData(self.ds, verbose=True)
        
        epoch_count = 0
        while epoch_count < 1000:
            epoch_count += 10
            self.trainer.trainUntilConvergence(dataset=self.ds, maxEpochs=10)
            networkwriter.NetworkWriter.writeToFile(self.net,'autosave.network')
        
        print 'MSE after', self.trainer.testOnData(ds, verbose=True)
        print ("\n")
        print 'Total epochs:', self.trainer.totalepochs

    def on_gui_change(self,x,s_index):
            try:
                if (self.compute == 0):
                    self.data_output[s_index] = float(x)
                    self.l_outputs[s_index].update(float(x))
            except:
                print ("WTF ? On Gui Change Error!")
                raise

    def main_loop(self):

        self.learnMapperDevice.poll(1)

        if ((self.learning == 1) and (self.compute == 0)):
            print ("Inputs: ")
            print (tuple(self.data_input.values()))
            print ("Outputs: ")
            print (tuple(self.data_output.values()))
            self.ds.addSample(tuple(self.data_input.values()),tuple(self.data_output.values()))  
        
        if ((self.compute == 1) and (self.learning == 0)):

            activated_out = self.net.activate(tuple(self.data_input.values()))
            for out_index in range(self.num_outputs):
                self.data_output[out_index] = activated_out[out_index]
                sliders[out_index].set(activated_out[out_index])
                self.l_outputs[out_index].update(self.data_output[out_index])

####################################################################################################################################


class PyImpUI(QWidget):
    
    def __init__(self):
        super(PyImpUI, self).__init__()
        self.CurrentNetwork = PyImpNetwork()
        self.initUI()

        timer = QTimer(self)
        self.connect(timer, SIGNAL("timeout()"), self.update)
        timer.start(1000)
        
    def initUI(self):

        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        widgets = self.findChildren(QWidget)
        #print "WIDGETS", widgets

        #Load UI created in QT Designer
        self.loadCustomWidget("PyImpMainWindow.ui")

        self.loadDataButton = self.findChild(QWidget,"loadDataButton")
        self.saveDataButton = self.findChild(QWidget,"saveDataButton")
        self.loadMappingButton = self.findChild(QWidget,"loadMappingButton")
        self.saveMappingButton = self.findChild(QWidget,"saveMappingButton")

        self.getDataButton = self.findChild(QWidget,"getDataButton")
        self.trainMappingButton = self.findChild(QWidget,"trainMappingButton")
        self.processOutputButton = self.findChild(QWidget,"processOutputButton")
        self.resetClassifierButton = self.findChild(QWidget,"resetClassifierButton")
        self.clearDataButton = self.findChild(QWidget,"clearDataButton")

        self.middleLayerEnable = self.findChild(QWidget,"middleLayerEnable")

        self.setSlidersButton = self.findChild(QWidget,"setSlidersButton")
        self.setSlidersButton.hide()

        self.chooseClassifier = self.findChild(QWidget,"chooseClassifierComboBox")

        # Activate the Buttons in the Initial Screen
        self.loadDataButton.clicked.connect(self.loadQDataset)
        self.saveDataButton.clicked.connect(self.saveQDataset)
        self.loadMappingButton.clicked.connect(self.loadQNetwork)
        self.saveMappingButton.clicked.connect(self.saveQNetwork)
        self.getDataButton.clicked.connect(self.learnQCallback)
        self.trainMappingButton.clicked.connect(self.trainQCallback)
        self.resetClassifierButton.clicked.connect(self.clearQNetwork)
        self.clearDataButton.clicked.connect(self.clearQDataSet)
        self.processOutputButton.clicked.connect(self.computeQCallback)

        self.middleLayerEnable.toggle()
        self.middleLayerEnable.stateChanged.connect(self.enableSliders)
        self.middleLayerEnable.setCheckState(Qt.Unchecked)

        self.show()

    def update(self):
        self.CurrentNetwork.learnMapperDevice.poll(1)
        self.CurrentNetwork.main_loop()

    def enableSliders(self,state):

        if state == Qt.Checked:
            #print "Middle Sliders Now Enabled"
            self.setSlidersButton.show()
            self.setSlidersButton.clicked.connect(self.on_setSlidersButton_clicked)

        else:
            #print "Middle Sliders Now Disabled"
            self.setSlidersButton.hide()        

    def loadCustomWidget(self,UIfile):
        loader = QUiLoader()
        file_ui = QFile(UIfile)
        file_ui.open(QFile.ReadOnly)
        self.mainWidget = loader.load(file_ui, self)
        self.setWindowTitle("Implicit Mapper")   
        file_ui.close()

    def on_setSlidersButton_clicked(self):

        self.slidersWindow = QMainWindow()
        self.slidersWindow.setGeometry(300,200,500,400)
        self.slidersWindow.setWindowTitle("Middle Layer Values")

        self.chooseNSliders = QLineEdit()
        self.chooseNSliders.setGeometry(10,32,100,25)
        self.chooseNSliders.setParent(self.slidersWindow)
        
        self.setButton = QPushButton("OK")
        self.setButton.setGeometry(self.chooseNSliders.width()+self.chooseNSliders.x()+5,32,50,25)
        self.setButton.setDisabled(1)
        self.setButton.setParent(self.slidersWindow)

        self.layoutNo = QHBoxLayout()
        self.layoutNo.addWidget(self.chooseNSliders)
        self.layoutNo.addWidget(self.setButton)

        self.chooseNSlidersLabel = QLabel("Set The Number of Middle Sliders")
        self.chooseNSlidersLabel.setGeometry(10,10,300,25)
        self.chooseNSlidersLabel.setParent(self.slidersWindow)

        self.layoutTop = QVBoxLayout()
        self.layoutTop.addLayout(self.layoutNo)
        self.layoutTop.addWidget(self.chooseNSlidersLabel)

        self.layoutSliders = QVBoxLayout()
        self.layoutSliders.setSpacing(5)
        self.layoutSliders.addLayout(self.layoutTop)

        self.chooseNSliders.textChanged.connect(self.setNumQSliders)
        self.slidersWindow.show()

    def setNumQSliders(self):
        
        #print "Creating Sliders Now From Sender"
        self.NoSlides = int(self.chooseNSliders.text())
        self.setButton.setEnabled(1)
        self.setButton.clicked.connect(self.createQSliders)

    def createQSliders(self):
        
        self.slidersWindow.hide()

        num_outputs = self.NoSlides
        sliders={}

        # If number of sliders is re-entered 
        if (len(self.slidersWindow.findChildren(QSlider)) != 0):
            print len(self.slidersWindow.findChildren(QSlider))
            
            for key in sliders.iterkeys():
                sliders[key].setParent(None)
                self.layoutSliders.removeWidget(sliders[key])
                del sliders[key]
        
        for s_index in range(self.NoSlides):
            print range(self.NoSlides)

            sliders[s_index] = QSlider()
            if s_index == 0: 
                sliders[s_index].setGeometry(10,70,self.slidersWindow.width()-20,10)
            else: 
                sliders[s_index].setGeometry(10,sliders[s_index-1].y()+20,self.slidersWindow.width()-20,10)

            sliders[s_index].setObjectName("Slider%s"%s_index)
            sliders[s_index].setOrientation(Qt.Horizontal)
            sliders[s_index].setRange(0,100)
            sliders[s_index].setParent(self.slidersWindow)
            self.layoutSliders.addWidget(sliders[s_index])

        self.slidersWindow.show()
        self.setButton.setDisabled(1)

    def loadQDataset(self):

        # Create Dialog to load the dataset from the directory
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
        loadDialog.setWindowTitle("Load Dataset")
        loadDialog.show()

        filename = loadDialog.getOpenFileName()
        PyImpNetwork.load_dataset(self.CurrentNetwork,filename)

    def saveQDataset(self):

        # Create Dialog to save the file in directory
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setWindowTitle("Save Dataset")
        saveDialog.show()

        filename = saveDialog.getSaveFileName()
        PyImpNetwork.save_dataset(self.CurrentNetwork,filename)
    
    def clearQDataSet(self):
        PyImpNetwork.clear_dataset(self.CurrentNetwork)

    def loadQNetwork(self):

        # Create Dialog to load the dataset from the directory
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
        loadDialog.setWindowTitle("Load Network")
        loadDialog.show()

        filename = loadDialog.getOpenFileName()
        PyImpNetwork.load_dataset(self.CurrentNetwork)

    def saveQNetwork(self):
        # Create Dialog to save the file in directory
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setWindowTitle("Save Network")
        saveDialog.show()

        filename = saveDialog.getSaveFileName()
        PyImpNetwork.save_net(self.CurrentNetwork)

    def clearQNetwork(self):
        PyImpNetwork.clear_network(self.CurrentNetwork)

    def learnQCallback(self):

        PyImpNetwork.learn_callback(self.CurrentNetwork)

        if self.CurrentNetwork.learning == 1:
            self.getDataButton.setDown(1)
            self.getDataButton.setText("Data ON")

        elif self.CurrentNetwork.learning == 0:
            self.getDataButton.setDown(0)
            self.getDataButton.setText("Get Data")

    def trainQCallback(self):
        PyImpNetwork.train_callback(self.CurrentNetwork)

    def computeQCallback(self):
        PyImpNetwork.compute_callback(self.CurrentNetwork)

        if self.CurrentNetwork.compute==1:
            self.processOutputButton.setDown(1)
            self.processOutputButton.setText("Computing Results ON")

        elif self.CurrentNetwork.compute ==0:
            self.processOutputButton.setDown(0)
            self.processOutputButton.setText("Process Results")

####################################################################################################################################################
####################################################################################################################################################

def main():

    # Run GUI Application
    app = QApplication(sys.argv)
    ex = PyImpUI()

    #Obtain Initial Number of Inputs and for Device
    if (len(sys.argv) == 4):
        try:
            ex.CurrentNetwork.setNumInputs(int(sys.argv[1]))
            ex.CurrentNetwork.setNumHiddenNodes(int(sys.argv[2]))
            ex.CurrentNetwork.setNumeOutputs(int(sys.argv[3]))
        except:
            print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
            sys.exit(1)

    elif (len(sys.argv) == 5):
        try:
            ex.CurrentNetwork.setNumInputs(int(sys.argv[1]))
            ex.CurrentNetwork.setNumHiddenNodes(int(sys.argv[2]))
            ex.CurrentNetwork.setNumeOutputs(int(sys.argv[3]))
            ex.CurrentNetwork.setReccurentFlag(int(sys.argv[4]))
        except:
            print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs, R/F == Recurrent/Feedforward Network)")
            sys.exit(1)

    elif (len(sys.argv) > 1):
            print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
            sys.exit(1)

    else:
        ex.CurrentNetwork.setNumInputs(8)
        ex.CurrentNetwork.setNumHiddenNodes(5)
        ex.CurrentNetwork.setNumeOutputs(8)
        print "No Input Arguments, setting defaults - 8 5 8"       

    print ("Input Arguments (#inputs, #hidden nodes, #outputs): " + str(ex.CurrentNetwork.num_inputs) + ", " + str(ex.CurrentNetwork.num_hidden) + ", " + str(ex.CurrentNetwork.num_outputs))        
    
    ex.CurrentNetwork.createMapperInputs(ex.CurrentNetwork.num_inputs)
    ex.CurrentNetwork.createMapperOutputs(ex.CurrentNetwork.num_outputs)
    ex.CurrentNetwork.createANN(ex.CurrentNetwork.num_inputs,ex.CurrentNetwork.num_hidden,ex.CurrentNetwork.num_outputs)

    sys.exit(app.exec_())

##################################################################################################################################################
# Run Main Function
if __name__ == '__main__':
    main()





