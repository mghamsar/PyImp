import sys
import mapper
import time

import PySide
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtWebKit import *
from PySide.QtUiTools import *
from learn_gest_network import PyImpNetwork

class pyimp_ui(QWidget):
    
    def __init__(self,training_network):
        super(pyimp_ui, self).__init__()
        
        self.current_network = training_network

        # Maintain a list of created widgets for the remove buttons
        self.button_list = []

        #Create a list of Grid Positions for the Edit Snapshots Window
        self.pos_list = []

        self.dsNumber = QLabel()

        self.initUI()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.QUpdate) 
        self.timer.start(50)
        
    def initUI(self):

        self.setStyle(QStyleFactory.create('Cleanlooks'))

        #Load UI created in QT Designer
        self.loadCustomWidget("PyImpMainWindowSnapShot.ui")
        self.setAutoFillBackground(1)

        widgets = self.findChildren(QWidget)
        #print "WIDGETS", widgets

        # Button Widgets in the Main Interface
        self.loadDataButton = self.findChild(QWidget,"loadDataButton")
        # self.loadDataButton.setColor("#FFC673")
        self.saveDataButton = self.findChild(QWidget,"saveDataButton")
        self.loadMappingButton = self.findChild(QWidget,"loadMappingButton")
        self.saveMappingButton = self.findChild(QWidget,"saveMappingButton")

        self.getDataButton = self.findChild(QWidget,"getDataButton")
        self.trainMappingButton = self.findChild(QWidget,"trainMappingButton")
        self.resetClassifierButton = self.findChild(QWidget,"resetClassifierButton")
        self.clearDataButton = self.findChild(QWidget,"clearDataButton")

        self.processOutputButton = self.findChild(QWidget,"processOutputButton")
        self.processOutputButton.setCheckable(True)

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
        print self.midLabel
        self.midLabel.hide()

        self.processResultsText = self.findChild(QLabel, "processResultsText")
        
        # Activate the Buttons in the Main Interface
        self.loadDataButton.clicked.connect(self.loadQDataset)
        self.saveDataButton.clicked.connect(self.saveQDataset)
        self.loadMappingButton.clicked.connect(self.loadQNetwork)
        self.saveMappingButton.clicked.connect(self.saveQNetwork)
        self.getDataButton.clicked.connect(self.learnQCallback)
        self.trainMappingButton.clicked.connect(self.trainQCallback)
        self.resetClassifierButton.clicked.connect(self.clearQNetwork)
        self.clearDataButton.clicked.connect(self.clearQDataSet)
        self.processOutputButton.clicked[bool].connect(self.computeQCallback)
        self.editSnapshots.clicked.connect(self.openEditSnapshotsWindow)

        self.middleLayerEnable.toggle()
        self.middleLayerEnable.stateChanged.connect(self.enableSliders)
        self.middleLayerEnable.setCheckState(Qt.Unchecked)
        
        self.snapshotWindow = QMainWindow()

        self.show()

    def QUpdate(self):
        self.current_network.learnMapperDevice.poll(0)
        self.current_network.update()
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
        self.layoutSliders.secontSpacing(5)
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
            sliders[s_index].setSliderPosition(5)
            self.current_network.data_middle["Slider%s"%s_index] = 5

            self.layoutSliders.addWidget(sliders[s_index])

        self.slidersWindow.show()
        self.setButton.setDisabled(1)
    
    def getSliderValue(self):
        sender = self.sender()
        sender_name = sender.objectName()

        self.current_network.data_middle[sender_name] = sender.value()
        #print "Middle Slider Values", self.current_network.data_middle.values()

    def loadQDataset(self):

        # Create Dialog to load the dataset from the directory
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
        loadDialog.setWindowTitle("Load Dataset")
        loadDialog.show()

        filename = loadDialog.getOpenFileName()
        self.current_network.load_dataset(filename)

    def saveQDataset(self):

        # Create Dialog to save the file in directory
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setWindowTitle("Save Dataset")
        saveDialog.show()

        filename = saveDialog.getSaveFileName()
        PyImpNetwork.save_dataset(self.current_network,filename)
    
    def clearQDataSet(self):
        self.current_network.clear_dataset()
        self.numberOfSnapshots.setText(str(len(self.current_network.temp_ds.keys())))

    def loadQNetwork(self):

        # Create Dialog to load the dataset from the directory
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
        loadDialog.setWindowTitle("Load Network")
        loadDialog.show()

        filename = loadDialog.getOpenFileName()
        self.current_network.load_dataset()

    def saveQNetwork(self):
        # Create Dialog to save the file in directory
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setWindowTitle("Save Network")
        saveDialog.show()

        filename = saveDialog.getSaveFileName()
        self.current_network.save_net()

    def clearQNetwork(self):
        # clear the previously calculated weights and start over
        self.current_network.clear_network()

    def learnQCallback(self):

        self.current_network.learn_callback()
        
        self.numberOfSnapshots.setText(str(self.current_network.snapshot_count))
        self.dsNumber.setText(str(self.current_network.snapshot_count))

        # Create the buttons in the edit snapshots screen 
        s_button = QPushButton("Remove Snapshot %s"%self.current_network.snapshot_count)
        s_button.resize(140,20)
        s_button.setObjectName("Dataset%d"%self.current_network.snapshot_count)
        s_button.setStyleSheet("QWidget {background-color:#DAFDE0;}")
        s_button.setParent(self.snapshotWindow)

        #This list contains the actual QWidget QPushButtons
        print "Button Added, total length", len(self.button_list)
        self.button_list.append(s_button)

        # Update the Grid positions for the list of buttons
        self.pos_list.append((self.current_network.snapshot_count/3,self.current_network.snapshot_count%3))
        print self.pos_list

        if self.current_network.learning == 1:
            self.getDataButton.setDown(1)
            self.getDataButton.setText("Taking Snapshot")

        elif self.current_network.learning == 0:
            self.getDataButton.setDown(0)
            self.getDataButton.setText("Snapshot")

    def trainQCallback(self):
        self.current_network.train_callback()

    def computeQCallback(self,pressed):

        if pressed: #self.processOutputButton.isChecked() == 1:
            print "Processing Output Now"
            self.processResultsText.setText("Computing Results is ON")
            self.current_network.compute = 1
            #self.current_network.compute_callback()

        else:
            print "Process output stopped"
            self.processResultsText.setText("Click to Compute Results")
            self.current_network.compute = 0


    def openEditSnapshotsWindow(self):

        self.addtoDsButton = QPushButton("Update Dataset")
        self.addtoDsButton.setGeometry(320,350,170,40)
        self.addtoDsButton.setStyleSheet("QWidget { background-color:#3AD76F;}")
        self.addtoDsButton.setParent(self.snapshotWindow)
        self.addtoDsButton.clicked.connect(self.updateQDataSet)

        self.dsLabel = QLabel("Number of Single Sets in Database:")
        self.dsLabel.setGeometry(30,350,270,40)
        self.dsLabel.setParent(self.snapshotWindow)
        
        self.dsNumber.setGeometry(270,350,100,40)
        self.dsNumber.setText(str(self.current_network.snapshot_count))
        self.dsNumber.setParent(self.snapshotWindow)
        
        self.snapshotGrid = QGridLayout()
        self.snapshotGrid.setHorizontalSpacing(10)
        self.snapshotGrid.setVerticalSpacing(10)

        #Display labels on Grid
        j = 0
        for button in self.button_list:
            button.setParent(self.snapshotWindow)
            self.snapshotGrid.addWidget(button,self.pos_list[j][0],self.pos_list[j][1],Qt.AlignCenter)
            button.move((self.pos_list[j][1])*(button.width()+5)+10,(self.pos_list[j][0])*(button.height()+10)+10)
            button.clicked.connect(self.removeTempDataSet)
            j = j+1
        
        self.snapshotWindow.setLayout(self.snapshotGrid)
        self.snapshotWindow.setGeometry(300,200,550,400)
        self.snapshotWindow.setWindowTitle("Edit Existing Snapshots")
        self.snapshotWindow.show()
    
    def updateQDataSet(self):
        self.current_network.update_ds()
        self.dsNumber.setText(str(self.current_network.snapshot_count))

    def removeTempDataSet(self):

        sender = self.sender()
        sender_name = sender.objectName()
        sender_id = sender_name.split("Dataset")
        sender_id = int(sender_id[1])
        print "Sender ID", sender_id
        
        self.current_network.remove_tempds(sender_id)
        print "Number of Items", self.snapshotGrid.count(), range(1,self.snapshotGrid.count()+1)

        sender.setParent(None)
        for button in sorted(self.button_list):
            if button.objectName() == sender.objectName():
                print "Found button to remove"
                self.button_list.remove(button)

        self.dsNumber.setText(str(self.current_network.snapshot_count))
        self.numberOfSnapshots.setText(str(self.current_network.snapshot_count))
        self.snapshotWindow.update()

    ############################################## Graph Drawing Methods Here #####################################################

    def paintEvent(self, event):
        self.qp = QPainter()
        self.qp.begin(self)
        self.qp.setRenderHint(QPainter.Antialiasing)
        self.paintSignals()
        self.qp.end()        

    # # Paint a single bar as part of a bar-graph
    def paintBar(self,x,y,barwidth,barheight):
        brush = QBrush(QColor("#9D0D02"),Qt.SolidPattern)
        rect = QRect(x,y,barwidth,barheight)
        self.qp.setBrush(brush)
        self.qp.drawRect(rect)

    # This function plots the individual signals coming into implicit mapper from both the input and the output
    def paintSignals(self):

        # # Overall Rectangle
        #brush1 = QBrush(QColor("#FFDE99"),Qt.Dense3Pattern)
        #self.qp.setBrush(brush1)
        self.qp.drawRect(self.inputPlot.x(),self.inputPlot.y(),self.outputPlot.width()*3+20,self.outputPlot.height())
        # self.qp.drawLine(self.inputPlot.x(),self.inputPlot.y(),self.outputPlot.x()+self.outputPlot.width(),self.outputPlot.y())
        # self.qp.drawLine(self.inputPlot.x(),self.inputPlot.y()+self.inputPlot.height(),self.outputPlot.x()+self.outputPlot.width(),self.outputPlot.y()+self.outputPlot.height())
        # self.qp.drawLine(self.inputPlot.x(),self.inputPlot.y(),self.inputPlot.x(),self.inputPlot.y()+self.inputPlot.height())
        # self.qp.drawLine(self.outputPlot.x()+self.outputPlot.width(),self.outputPlot.y(),self.outputPlot.x()+self.outputPlot.width(),self.outputPlot.y()+self.outputPlot.height())

        # Input Plot Background
        # self.inputRect = QRect(self.inputPlot.x(),self.inputPlot.y(),self.inputPlot.width(), self.inputPlot.height())
        #brush1 = QBrush(QColor("#FFDE99"),Qt.Dense3Pattern)
        #self.qp.setBrush(brush1)
        #self.qp.drawRect(20,65,300,220)
        #self.qp.drawRect(330,15,490,180)

        # # Middle Plot Background
        # self.middleRect = QRect(self.middlePlot.x(),self.middlePlot.y(),self.middlePlot.width(),self.middlePlot.height())
        # brush = QBrush(QColor("#FFDE99"),Qt.Dense3Pattern)
        # self.qp.setBrush(brusneth)
        # self.qp.drawRect(self.middleRect)

        # # Output Plot Background
        # self.outputRect = QRect(self.outputPlot.x(),self.outputPlot.y(),self.outputPlot.width(),self.outputPlot.height())
        # brush2 = QBrush(QColor("#FFDE99"),Qt.Dense3Pattern)
        # self.qp.setBrush(brush2)
        # self.qp.drawRect(self.outputRect)

        # Input Bars
        if len(self.current_network.data_input.keys())>1:
            barwidth_in = float(self.inputPlot.width())/len(self.current_network.data_input.keys())-5
        else: 
            barwidth_in = 1
        cnt = 0
        for inputsig, sigvalue in sorted(self.current_network.data_input.iteritems()):
            #print "input rectangle %s"%inputsig, sigvalue
            sigmax = 1
            if (sigvalue > sigmax): 
                sigmax = sigvalue

            sigvalue = (sigvalue/sigmax)
            self.paintBar(self.inputPlot.x()+10+cnt*barwidth_in,self.inputPlot.y() + self.inputPlot.height(),barwidth_in,(-1)*abs(sigvalue*self.inputPlot.height()))
            cnt = cnt+1

        # Output Bars
        if len(self.current_network.data_output.keys())>1:
            barwidth_out = self.outputPlot.width()/len(self.current_network.data_output.keys())-5
        else: 
            barwidth_out = 1
        cnt2 = 0
        for outputsig, outvalue in sorted(self.current_network.data_output.iteritems()):
            #print "output rectangle %s"%outputsig, outvalue
            sigmax2 = 1
            if (outvalue > sigmax2): 
                sigmax2 = outvalue
            
            outvalue = (outvalue/sigmax2)
            self.paintBar(self.outputPlot.x()+10+cnt2*barwidth_out,self.outputPlot.y() + self.outputPlot.height(),barwidth_out,(-1)*abs(outvalue*self.outputPlot.height()))
            cnt2 = cnt2+1

        # Middle Bars
        if len(self.current_network.data_middle.keys())>=1: 
            barwidth_mid = self.middlePlot.width()/len(self.current_network.data_middle.keys())-5
            cnt3 = 0 
            for midsig, midval in sorted(self.current_network.data_middle.iteritems()):
                #print "output rectangle %s"%outputsig, outvalue
                # if (midval > sigmax2): 
                #     sigmax2 = outvalue
                # outvalue = (outvalue/sigmax2)
                self.paintBar(self.middlePlot.x()+10+cnt3*barwidth_mid,self.middlePlot.y() + self.middlePlot.height(),barwidth_mid,(-1)*abs(midval))
                cnt3 = cnt3+1

def main():
    # Run GUI Application
    import sys
    print sys.argv
    app = QApplication(sys.argv)
    net = PyImpNetwork()
    print net
    ex = pyimp_ui(net)

    #Obtain Initial Number of Inputs and for Device
    if (len(sys.argv) == 4):
        try:
            net.setNumInputs(int(sys.argv[1]))
            net.setNumHiddenNodes(int(sys.argv[2]))
            net.setNumeOutputs(int(sys.argv[3]))
        except:
            print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
            sys.exit(1)

    elif (len(sys.argv) == 5):
        try:
            net.setNumInputs(int(sys.argv[1]))
            net.setNumHiddenNodes(int(sys.argv[2]))
            net.setNumeOutputs(int(sys.argv[3]))
            net.setReccurentFlag(int(sys.argv[4]))
        except:
            print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs, R/F == Recurrent/Feedforward Network)")
            sys.exit(1)

    elif (len(sys.argv) > 1):
            print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
            sys.exit(1)

    else:
        net.setNumInputs(8)
        net.setNumHiddenNodes(5)
        net.setNumeOutputs(8)
        print "No Input Arguments, setting defaults - 8 5 8"       

    print ("Input Arguments (#inputs, #hidden nodes, #outputs): " + str(net.num_inputs) + ", " + str(net.num_hidden) + ", " + str(net.num_outputs))        
    
    net.createMapperInputs(net.num_inputs)
    net.createMapperOutputs(net.num_outputs)
    net.createANN(net.num_inputs,net.num_hidden,net.num_outputs)

    sys.exit(app.exec_())