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
        self.data_middle = {}

        # store a list of the signals obtained from the device
        self.input_names = []
        self.output_names = []

        #temporary array for storying single snapshots
        self.temp_ds = {}
        self.snapshot_count = 0

        self.learnMapperDevice = mapper.device("Implicit_LearnMapper",9002)

    # mapper signal handler (updates self.data_input[sig_indx]=new_float_value)
    def h(self,sig, f):
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
            print ("creating input", "/in"+str(l_num))

        # Set initial Data Input values for Network to 0
        for s_index in range(n_inputs):
            self.data_input[s_index] = 0.0

    def createMapperOutputs(self,n_outputs):
        #create mapper signals (n_outputs)
        for l_num in range(n_outputs):
            self.l_outputs[l_num] = self.learnMapperDevice.add_output("/out%d"%l_num, 1, 'f',None,0.0,1.0)
            self.l_outputs[l_num].set_query_callback(self.h)
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

        self.update_output()

        # Save data to a temporary database in case they need to be edited before adding to the Supervised Dataset
        self.snapshot_count = self.snapshot_count+1
        self.temp_ds[self.snapshot_count] = {}
        self.temp_ds[self.snapshot_count]["input"] = tuple(self.data_input.values())
        self.temp_ds[self.snapshot_count]["output"] = tuple(self.data_output.values())

        print "Values before going to temp_ds", self.data_input.values(), "   ", self.data_output.values()
        print self.snapshot_count, "(Input, Output)", self.temp_ds[self.snapshot_count]

        self.update_ds()

    def remove_tempds(self,objectNum):

        if objectNum in self.temp_ds.iterkeys():
            print "Found DS to delete", objectNum
            del self.temp_ds[objectNum]

        else: 
            print "Error, This database entry does not exist"

    def compute_callback(self):

        activated_out = self.net.activate(tuple(self.data_input.values()))

        for out_index in range(self.num_outputs):
            self.data_output[out_index] = activated_out[out_index]
            self.l_outputs[out_index].update(self.data_output[out_index])
            # TO DO: Update Outputs Plot dynamically with new values

    def train_callback(self):
        self.trainer = BackpropTrainer(self.net, learningrate=0.01, lrdecay=1, momentum=0.0, verbose=True)
        
        print 'MSE before', self.trainer.testOnData(self.ds, verbose=True)
        epoch_count = 0
        while epoch_count < 1000:
            epoch_count += 10
            self.trainer.trainUntilConvergence(dataset=self.ds, maxEpochs=10)
            networkwriter.NetworkWriter.writeToFile(self.net,'autosave.network')
        
        print 'MSE after', self.trainer.testOnData(self.ds, verbose=True)
        print ("\n")
        print 'Total epochs:', self.trainer.totalepochs

    def update_output(self):
        # Get Value from the output by sending a query request
        for index in range(self.num_outputs):
            self.data_output[index] = self.l_outputs[index].query_remote()
        
        print self.data_output.values()

    def update_ds(self):
        for key in self.temp_ds.iterkeys(): 
            self.ds.addSample(self.temp_ds[key]["input"],self.temp_ds[key]["output"])

####################################################################################################################################

class PyImpUI(QWidget):
    
    def __init__(self):
        super(PyImpUI, self).__init__()
        self.CurrentNetwork = PyImpNetwork()
        self.initUI()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.QUpdate) 
        # self.connect(timer, SIGNAL("timeout()"), )
        self.timer.start(1000)
        
    def initUI(self):

        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        widgets = self.findChildren(QWidget)
        #print "WIDGETS", widgets

        #Load UI created in QT Designer
        self.loadCustomWidget("PyImpMainWindowSnapShot.ui")

        # Button Widgets in the Main Interface
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

        self.numberOfSnapshots = self.findChild(QLabel,"noSnapshots")
        self.editSnapshots = self.findChild(QWidget,"editSnapshots")

        #Graphics Views for the Signals
        self.inputPlot = self.findChild(QWidget,"inputSignals")
        self.outputPlot = self.findChild(QWidget,"outputSignals")
        self.middlePlot = self.findChild(QWidget,"middleSignals")
        self.middlePlot.hide()

        self.midLabel = self.findChild(QLabel,"midlabel")
        self.midLabel.hide()

        # Activate the Buttons in the Main Interface
        self.loadDataButton.clicked.connect(self.loadQDataset)
        self.saveDataButton.clicked.connect(self.saveQDataset)
        self.loadMappingButton.clicked.connect(self.loadQNetwork)
        self.saveMappingButton.clicked.connect(self.saveQNetwork)
        self.getDataButton.clicked.connect(self.learnQCallback)
        self.trainMappingButton.clicked.connect(self.trainQCallback)
        self.resetClassifierButton.clicked.connect(self.clearQNetwork)
        self.clearDataButton.clicked.connect(self.clearQDataSet)
        self.processOutputButton.clicked.connect(self.computeQCallback)
        self.editSnapshots.clicked.connect(self.openEditSnapshotsWindow)

        self.middleLayerEnable.toggle()
        self.middleLayerEnable.stateChanged.connect(self.enableSliders)
        self.middleLayerEnable.setCheckState(Qt.Unchecked)

        self.show()

    def QUpdate(self):
        self.CurrentNetwork.learnMapperDevice.poll(1)
        #self.CurrentNetwork.update_output()
        self.update()

    def loadCustomWidget(self,UIfile):
        loader = QUiLoader()
        file_ui = QFile(UIfile)
        file_ui.open(QFile.ReadOnly)
        self.mainWidget = loader.load(file_ui, self)
        self.setWindowTitle("Implicit Mapper")   
        file_ui.close()

    def enableSliders(self,state):

        if state == Qt.Checked:
            #print "Middle Sliders Now Enabled"
            self.setSlidersButton.show()
            self.midLabel.show()
            self.middlePlot.show()
            self.setSlidersButton.clicked.connect(self.openEditSlidersWindow)

        else:
            #print "Middle Sliders Now Disabled"
            self.setSlidersButton.hide() 
            self.middlePlot.hide()
            self.midLabel.hide()

    def openEditSlidersWindow(self):

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
        self.NoSlides = int(self.chooseNSliders.text())
        self.setButton.setEnabled(1)
        self.setButton.clicked.connect(self.createQSliders)

    def createQSliders(self):
        
        self.slidersWindow.hide()

        num_outputs = self.NoSlides
        sliders={}

        # If number of sliders is re-entered 
        if (len(self.slidersWindow.findChildren(QSlider)) != 0):
            #print len(self.slidersWindow.findChildren(QSlider))
            
            for key in sliders.iterkeys():
                sliders[key].setParent(None)
                self.layoutSliders.removeWidget(sliders[key])
                del sliders[key]
        
        for s_index in range(self.NoSlides):
            #print range(self.NoSlides)

            sliders[s_index] = QSlider()
            if s_index == 0: 
                sliders[s_index].setGeometry(10,70,self.slidersWindow.width()-20,10)
            else: 
                sliders[s_index].setGeometry(10,sliders[s_index-1].y()+20,self.slidersWindow.width()-20,10)

            sliders[s_index].setObjectName("Slider%s"%s_index)
            sliders[s_index].setOrientation(Qt.Horizontal)
            sliders[s_index].setRange(0,100)
            sliders[s_index].setParent(self.slidersWindow)
            sliders[s_index].valueChanged.connect(self.getSliderValue)
            self.layoutSliders.addWidget(sliders[s_index])

        self.slidersWindow.show()
        self.setButton.setDisabled(1)
    
    def getSliderValue(self):
        sender = self.sender()
        sender_name = sender.objectName()

        self.CurrentNetwork.data_middle[sender_name] = sender.value()
        #print "Middle Slider Values", self.CurrentNetwork.data_middle.values()

    def loadQDataset(self):

        # Create Dialog to load the dataset from the directory
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
        loadDialog.setWindowTitle("Load Dataset")
        loadDialog.show()

        filename = loadDialog.getOpenFileName()
        self.CurrentNetwork.load_dataset(filename)

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
        self.CurrentNetwork.clear_dataset()
        self.numberOfSnapshots.setText(str(self.CurrentNetwork.snapshot_count))

    def loadQNetwork(self):

        # Create Dialog to load the dataset from the directory
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
        loadDialog.setWindowTitle("Load Network")
        loadDialog.show()

        filename = loadDialog.getOpenFileName()
        self.CurrentNetwork.load_dataset()

    def saveQNetwork(self):
        # Create Dialog to save the file in directory
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setWindowTitle("Save Network")
        saveDialog.show()

        filename = saveDialog.getSaveFileName()
        self.CurrentNetwork.save_net()

    def clearQNetwork(self):
        self.CurrentNetwork.clear_network()

    def learnQCallback(self):

        self.CurrentNetwork.learn_callback()
        cur_count = int(self.numberOfSnapshots.text())
        self.numberOfSnapshots.setText(str(cur_count+1))

        if self.CurrentNetwork.learning == 1:
            self.getDataButton.setDown(1)
            self.getDataButton.setText("Taking Snapshot")

        elif self.CurrentNetwork.learning == 0:
            self.getDataButton.setDown(0)
            self.getDataButton.setText("Snapshot")

    def trainQCallback(self):
        self.CurrentNetwork.train_callback()

    def computeQCallback(self):
        print "Processing Output Now"
        self.CurrentNetwork.compute_callback()

        if self.CurrentNetwork.compute==1:
            self.processOutputButton.setDown(1)
            self.processOutputButton.setText("Computing Results ON")

        elif self.CurrentNetwork.compute ==0:
            self.processOutputButton.setDown(0)
            self.processOutputButton.setText("Process Results")

    def openEditSnapshotsWindow(self):

        self.snapshotWindow = QMainWindow()

        self.addtoDsButton = QPushButton("Update Dataset")
        self.addtoDsButton.setGeometry(320,350,170,40)
        self.addtoDsButton.setParent(self.snapshotWindow)
        self.addtoDsButton.clicked.connect(self.updateQDataSet)
        
        self.snapshotGrid = QGridLayout()
        self.snapshotGrid.setHorizontalSpacing(10)
        self.snapshotGrid.setVerticalSpacing(10)

        # Maintain a list of created widgets
        button_list = []
        label_list = []

        #List of Grid Positions
        pos = []

        for s, val in self.CurrentNetwork.temp_ds.iteritems():
            s_button = QPushButton("Remove")
            s_label = QLabel("Snapshot %s"%s)
            s_button.resize(80,20)
            s_label.resize(80,30)
            s_button.setObjectName("Dataset%d"%s)
            s_label.setObjectName("LabelDataset%d"%s)
            s_label.setParent(self.snapshotWindow)
            s_button.setParent(self.snapshotWindow)

            #These lists contain the actual QWidgets
            button_list.append(s_button)
            label_list.append(s_label)
            
            #Grid Positions
            p = s-1
            pos.append((p/5,p%5))

        #Display labels on Grid
        j = 0
        for label in label_list:
            label.setParent(self.snapshotWindow)
            label.setAlignment(Qt.AlignCenter)
            self.snapshotGrid.addWidget(label,pos[j][0],pos[j][1],Qt.AlignCenter)
            label.move((pos[j][1])*(label.width()+5)+10,(pos[j][0])*(label.height()+10)+10)
            j = j+1
        
        #Display buttons on Grid
        k = 0
        for button in button_list:
            button.setParent(self.snapshotWindow)
            self.snapshotGrid.addWidget(button,pos[k][0]+1,pos[k][1],Qt.AlignCenter)
            button.move((pos[k][1])*(button.width()+5)+10,(pos[k][0])*(button.height()+20)+35)
            button.clicked.connect(self.removeTempDataSet)
            k = k+1

        self.snapshotWindow.setLayout(self.snapshotGrid)
        self.snapshotWindow.setGeometry(300,200,500,400)
        self.snapshotWindow.setWindowTitle("Edit Existing Snapshots")
        self.snapshotWindow.show()
    
    def updateQDataSet(self):
        self.CurrentNetwork.update_ds()

    def removeTempDataSet(self):
        sender = self.sender()
        sender_name = sender.objectName()
        sender_id = sender_name.split("Dataset")
        sender_id = int(sender_id[1])
        print "Sender ID", sender_id
        
        self.CurrentNetwork.remove_tempds(sender_id)
        print "Number of Items", self.snapshotGrid.count(), range(1,self.snapshotGrid.count()+1)

        for i in range(1,self.snapshotGrid.count()+1):

            b = self.snapshotGrid.takeAt(i)
            b_widget = b.widget()
            print b_widget

            if b_widget.objectName() == str(sender_id):
                b.widget().setParent(None)
                b.widget().deleteLater()

        self.snapshotWindow.update()

        # b_label = self.snapshotWindow.findChild(QLabel,"LabelDataset%d"%sender_id)

    ############################################## Graph Drawing Methods Here #####################################################

    def paintEvent(self, event):
        self.qp = QPainter()
        self.qp.begin(self)
        self.qp.setRenderHint(QPainter.Antialiasing)
        self.paintSignals()
        self.qp.end()        

    # Paint a single bar as part of a bar-graph
    def paintBar(self,x,y,barwidth,barheight):

        brush = QBrush(QColor(0,100,160),Qt.SolidPattern)
        rect = QRect(x,y,barwidth,barheight)
        self.qp.setBrush(brush)
        self.qp.drawRect((rect.adjusted(0, 0, -1, -1)))

    # This function plots the individual signals coming into implicit mapper from both the input and the output
    def paintSignals(self):
        #print "Length of Input Signals", len(self.CurrentNetwork.data_input.keys())

        # Input Plot Background
        self.inputRect = QRect(self.inputPlot.x(),self.inputPlot.y(),self.inputPlot.width(), self.inputPlot.height())
        brush1 = QBrush(QColor(255,200,0),Qt.Dense3Pattern)
        self.qp.setBrush(brush1)
        self.qp.drawRect((self.inputRect.adjusted(0, 0, -1, -1)))

        # Middle Plot Background
        self.middleRect = QRect(self.middlePlot.x(),self.middlePlot.y(),self.middlePlot.width(),self.middlePlot.height())
        brush = QBrush(QColor(255,150,0),Qt.Dense3Pattern)
        self.qp.setBrush(brush)
        self.qp.drawRect((self.middleRect.adjusted(0, 0, -1, -1)))

        # Output Plot Background
        self.outputRect = QRect(self.outputPlot.x(),self.outputPlot.y(),self.outputPlot.width(),self.outputPlot.height())
        brush2 = QBrush(QColor(255,180,160),Qt.Dense3Pattern)
        self.qp.setBrush(brush2)
        self.qp.drawRect((self.outputRect.adjusted(0, 0, -1, -1)))
        
        # Input Bars 
        barwidth_in = self.inputPlot.width()/len(self.CurrentNetwork.data_input.keys())-5
        cnt = 0
        for inputsig, sigvalue in self.CurrentNetwork.data_input.iteritems():
            #print "input rectangle %s"%inputsig, sigvalue
            self.paintBar(self.inputPlot.x()+5+cnt*barwidth_in,self.inputPlot.y() + self.inputPlot.height(),barwidth_in,(-1)*abs(sigvalue*self.inputPlot.height()))
            cnt = cnt+1

        # Output Bars
        # print "Length of Ouptput Signals", len(self.CurrentNetwork.data_output.keys())
        barwidth_out = self.outputPlot.width()/len(self.CurrentNetwork.data_output.keys())-5
        cnt2 = 0 
        for outputsig, outvalue in self.CurrentNetwork.data_output.iteritems():
            #print "output rectangle %s"%outputsig, outvalue
            self.paintBar(self.outputPlot.x()+5+cnt2*barwidth_out,self.outputPlot.y() + self.outputPlot.height(),barwidth_out,(-1)*abs(outvalue*self.outputPlot.height()))
            cnt2 = cnt2+1

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





