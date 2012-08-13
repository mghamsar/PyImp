import sys
#import mapper

import pybrain
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer
#from pybrain.tools.customxml import networkwriter

import PySide
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtWebKit import *
from PySide.QtUiTools import *

import Tkinter
import re
import tkFileDialog

class PyImpNetwork():

    def __init__(self):
        self.ds = 0
        self.net = 0
        self.learning = 0
        self.compute = 0

    def load_dataset(self):
        open_filename = tkFileDialog.askopenfilename()
        self.ds=SupervisedDataSet.loadFromFile(open_filename)
        print self.ds

    def save_dataset(self):
        save_filename = tkFileDialog.asksaveasfilename()
        self.ds.saveToFile(save_filename)
        csv_file=open(save_filename+".csv",'w')
        csv_file.write("[inputs][outputs]\r\n")
        for inpt, tgt in ds:
                new_str=str("{" + repr(inpt) + "," + repr(tgt) + "}")
                new_str=new_str.strip('\n')
                new_str=new_str.strip('\r')
                new_str=new_str+"\r"
                #print(repr(new_str))
                csv_file.write(new_str)
        csv_file.close()

    def save_net(self):
        save_filename = tkFileDialog.asksaveasfilename()
        networkwriter.NetworkWriter.writeToFile(net,save_filename)

    def load_net(self):
        from pybrain.tools.customxml import networkreader
        open_filename = tkFileDialog.askopenfilename()
        self.net=networkreader.NetworkReader.readFrom(open_filename)

    def clear_dataset(self):
        if self.ds != 0:
            self.ds.clear()


    def clear_network(self):
        #resets the module buffers but doesn't reinitialise the connection weights
        #TODO: reinitialise network here or make a new option for it.
        self.net.reset()

    def learn_callback(self):

        if self.learning == 1:
            b_learn.config(relief='raised',text="Acquire Training Data (OFF)",bg='gray')
            self.learning=0
            print ("learning is now OFF")

        elif self.learning ==0:
            b_learn.config(relief='sunken',text="Acquiring Training Data (ON)",bg='red')
            self.learning=1
            print ("learning is now ON")

        print ("learning is", self.learning)

    def compute_callback(self):

        if self.compute==1:
            b_compute.config(relief='raised',text="Press to compute network outputs (OFF)",bg='gray')
            self.compute =0
            print ("Compute network output is now OFF!")
        elif self.compute ==0:
            b_compute.config(relief='sunken',text="Computing network outputs(ON)",bg='coral')
            self.compute =1
            print ("Compute network output is now ON!")

    def train_callback(self):
        trainer = BackpropTrainer(net, learningrate=0.01, lrdecay=1, momentum=0.0, verbose=True)
        
        print 'MSE before', trainer.testOnData(ds, verbose=True)
        epoch_count = 0
        while epoch_count < 1000:
            epoch_count += 10
            trainer.trainUntilConvergence(dataset=ds, maxEpochs=10)
            networkwriter.NetworkWriter.writeToFile(net,'autosave.network')
        print 'MSE after', trainer.testOnData(ds, verbose=True)
        print ("\n")
        print 'Total epochs:', trainer.totalepochs

    def on_gui_change(self,x,s_index):
            try:
                if (self.compute==0):
                    data_output[s_index]=float(x)
                    l_outputs[s_index].update(float(x))
            except:
                print ("WTF MATE? On Gui Change Error!")
                raise

    def main_loop(self):
        if ((self.learning==1) and (self.compute ==0)):
                    print ("Inputs: ")
                    print (tuple(data_input.values()))
                    print ("Outputs: ")
                    print (tuple( data_output.values()))
                    self.ds.addSample(tuple(data_input.values()),tuple(data_output.values()))        
        if (l_map.poll(1)) and ((self.compute==1) and (learning==0)):
                #print "inputs to net: ", data_input
                activated_out=net.activate(tuple(data_input.values()))
                #print "Activated outs: ", activated_out
                for out_index in range(num_outputs):
                    data_output[out_index]=activated_out[out_index]
                    sliders[out_index].set(activated_out[out_index])
                    l_outputs[out_index].update(data_output[out_index])
    
    def ontimer(self):
        main_loop()
        master.after(10, ontimer)

####################################################################################################################################


class PyImpUI(QWidget):
    
    def __init__(self):
        super(PyImpUI, self).__init__()

        self.CurrentNetwork = PyImpNetwork()

        self.initUI()
        
    def initUI(self):

        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        widgets = self.findChildren(QWidget)
        print "WIDGETS", widgets

        #Load UI created in QT Designer
        self.loadCustomWidget("PyImpMainWindow.ui")

        loadDataButton = self.findChild(QWidget,"loadDataButton")
        saveDataButton = self.findChild(QWidget,"saveDataButton")
        loadMappingButton = self.findChild(QWidget,"loadMappingButton")
        saveMappingButton = self.findChild(QWidget,"saveMappingButton")

        getDataButton = self.findChild(QWidget,"getDataButton")
        trainMappingButton = self.findChild(QWidget,"trainMappingButton")
        processOutputButton = self.findChild(QWidget,"processOutputButton")
        resetClassifierButton = self.findChild(QWidget,"resetClassifierButton")
        clearDataButton = self.findChild(QWidget,"clearDataButton")

        middleLayerEnable = self.findChild(QWidget,"middleLayerEnable")

        self.setSlidersButton = self.findChild(QWidget,"setSlidersButton")
        self.setSlidersButton.hide()

        chooseClassifier = self.findChild(QWidget,"chooseClassifierComboBox")

        # Activate the Buttons in the Initial Screen
        loadDataButton.clicked.connect(self.loadQDataset)
        saveDataButton.clicked.connect(self.saveQDataset)
        loadMappingButton.clicked.connect(self.loadQNetwork)
        saveMappingButton.clicked.connect(self.saveQNetwork)
        getDataButton.clicked.connect(self.learnQCallback)
        trainMappingButton.clicked.connect(self.trainQCallback)
        resetClassifierButton.clicked.connect(self.clearQNetwork)
        clearDataButton.clicked.connect(self.clearQDataSet)
        processOutputButton.clicked.connect(self.computeQCallback)

        middleLayerEnable.toggle()
        middleLayerEnable.stateChanged.connect(self.enableSliders)
        middleLayerEnable.setCheckState(Qt.Unchecked)

        self.show()

    def enableSliders(self,state):

        if state == Qt.Checked:
            print "Middle Sliders Now Enabled"
            self.setSlidersButton.show()
            self.setSlidersButton.clicked.connect(self.on_setSlidersButton_clicked)

        else:
            print "Middle Sliders Now Disabled"
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

        # create mapper signals (outputs)
        for l_num in range(num_outputs):
            l_outputs[l_num]=l_map.add_output("/out"+str(l_num), 1, 'f',None,0.0,1.0)
            l_inputs[l_num + num_inputs]=l_map.add_input("/out%d"%l_num, 1, 'f',None,0,1.0, h)
            #l_map.poll(0)
            print ("creating output","/out"+str(l_num))

        # self.mainloop()

    def loadQDataset(self):
        PyImpNetwork.load_dataset(self.CurrentNetwork)

    def saveQDataset(self):
        PyImpNetwork.save_dataset(self.CurrentNetwork)
    
    def clearQDataSet(self):
        PyImpNetwork.clear_dataset(self.CurrentNetwork)

    def loadQNetwork(self):
        PyImpNetwork.load_dataset(self.CurrentNetwork)

    def saveQNetwork(self):
        PyImpNetwork.save_net(self.CurrentNetwork)

    def clearQNetwork(self):
        PyImpNetwork.clear_network(self.CurrentNetwork)

    def learnQCallback(self):
        PyImpNetwork.learn_callback(self.CurrentNetwork)

    def trainQCallback(self):
        PyImpNetwork.train_callback(self.CurrentNetwork)

    def computeQCallback(self):
        PyImpNetwork.compute_callback(self.CurrentNetwork)
        

    #mapper signal handler (updates data_input[sig_indx]=new_float_value)
    def h(self,sig, f):
        try:
            global data_input
            global data_output
            
            if '/in' in sig.name:
                s_indx=str.split(sig.name,"/in")
                
                data_input[int(s_indx[1])]=float(f)
                #print(int(s_indx[1]),data_input[int(s_indx[1])])

            elif '/out' in sig.name:
                if (learning==1):
                    #print "test"
                    s_indx=str.split(sig.name,"/out")
                    data_output[int(s_indx[1])]=float(f)
                    #print(int(s_indx[1]),data_output[int(s_indx[1])])
        except:
            print "WTF, h handler not working"


def main():

    recurrent_flag=False; # default case is a nonrecurrent feedforward network

    # if (len(sys.argv)==4):
    #         #print (sys.argv)
    #         try:
    #                 num_inputs=int(sys.argv[1])
    #                 num_hidden=int(sys.argv[2])
    #                 num_outputs=int(sys.argv[3])
    #                 print ("Input Arguments (#inputs, #hidden nodes, #outputs): " + str(num_inputs) + ", " + str(num_hidden) + ", " + str(num_outputs) )        
    #         except:
    #                 print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
    #                 sys.exit(1)
    # elif (len(sys.argv)==5):
    #     try:
    #         num_inputs=int(sys.argv[1])
    #         num_hidden=int(sys.argv[2])
    #         num_outputs=int(sys.argv[3])
    #         if (sys.argv[4] == "R"):
    #             recurrent_flag=True
    #         elif (sys.argv[4] == "F"):
    #             recurrent_flag=False
    #         print ("Input Arguments (#inputs, #hidden nodes, #outputs): " + str(num_inputs) + ", " + str(num_hidden) + ", " + str(num_outputs) + ", recurrent = " + str(recurrent_flag))
    #     except:
    #         print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs, R/F == Recurrent/Feedforward)")
    #         sys.exit(1)
    # elif (len(sys.argv)>1):
    #         print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
    #         sys.exit(1)

    # else:
    #         #number of network inputs
    #         num_inputs=8
    #         #number of network outputs
    #         num_outputs=8
    #         #number of hidden nodes
    #         num_hidden=5
    #         print ("No Input Arguments (#inputs, #hidden nodes, #outputs), defaulting to: " + str(num_inputs) + ", " + str(num_hidden) + ", " + str(num_outputs) )        
    # #instatiate mapper
    # l_map=mapper.device("learn_mapper",9002)

    # l_inputs={}
    # l_outputs={}
    # data_input={}
    # data_output={}

    # for s_index in range(num_inputs):
    #     data_input[s_index]=0.0
    # for s_index in range (num_outputs):
    #     data_output[s_index]=0.0

    # Run GUI Application
    app = QApplication(sys.argv)
    ex = PyImpUI()

    # #create mapper signals (inputs)
    # for l_num in range(num_inputs):
    #     l_inputs[l_num]=l_map.add_input("/in%d"%l_num, 1, 'f',None,0,1.0, h)
    #     #l_map.poll(0)
    #     print ("creating input", "/in"+str(l_num))
        
    # #create mapper signals (outputs)
    # for l_num in range(num_outputs):
    #     l_outputs[l_num]=l_map.add_output("/out"+str(l_num), 1, 'f',None,0.0,1.0)
    #     l_inputs[l_num + num_inputs]=l_map.add_input("/out%d"%l_num, 1, 'f',None,0,1.0, h)
    #     #l_map.poll(0)
    #     print ("creating output","/out"+str(l_num))

    # #create network
    # net = buildNetwork(num_inputs,num_hidden,num_outputs,bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, recurrent=recurrent_flag)
    # #create dataSet
    # ds = SupervisedDataSet(num_inputs, num_outputs)

    # ontimer()

    sys.exit(app.exec_())

############################################################      
# Run Main Function
if __name__ == '__main__':
    main()





