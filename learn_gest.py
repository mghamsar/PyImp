from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer
from pybrain.tools.customxml import networkwriter
import mapper
import Tkinter
import re

import tkFileDialog
import sys

recurrent_flag=False; # default case is a nonrecurrent feedforward network

if (len(sys.argv)==4):
        #print (sys.argv)
        try:
                num_inputs=int(sys.argv[1])
                num_hidden=int(sys.argv[2])
                num_outputs=int(sys.argv[3])
                print ("Input Arguments (#inputs, #hidden nodes, #outputs): " + str(num_inputs) + ", " + str(num_hidden) + ", " + str(num_outputs) )        
        except:
                print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
                sys.exit(1)
elif (len(sys.argv)==5):
	try:
		num_inputs=int(sys.argv[1])
		num_hidden=int(sys.argv[2])
		num_outputs=int(sys.argv[3])
		if (sys.argv[4] == "R"):
			recurrent_flag=True
		elif (sys.argv[4] == "F"):
			recurrent_flag=False
		print ("Input Arguments (#inputs, #hidden nodes, #outputs): " + str(num_inputs) + ", " + str(num_hidden) + ", " + str(num_outputs) + ", recurrent = " + str(recurrent_flag))
	except:
		print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs, R/F == Recurrent/Feedforward)")
		sys.exit(1)
elif (len(sys.argv)>1):
        print ("Bad Input Arguments (#inputs, #hidden nodes, #outputs)")
        sys.exit(1)

else:
        #number of network inputs
        num_inputs=8
        #number of network outputs
        num_outputs=8
        #number of hidden nodes
        num_hidden=5
        print ("No Input Arguments (#inputs, #hidden nodes, #outputs), defaulting to: " + str(num_inputs) + ", " + str(num_hidden) + ", " + str(num_outputs) )        
#instatiate mapper
l_map=mapper.device("learn_mapper",9002)

l_inputs={}
l_outputs={}
data_input={}
data_output={}
learning = 0
compute = 0

for s_index in range(num_inputs):
	data_input[s_index]=0.0
#	data_input[s_index+10]=0.0
for s_index in range (num_outputs):
	data_output[s_index]=0.0


sliders={}

master=Tkinter.Tk()
master.title("PyBrain Mapper Demo")
master.resizable(height=True, width=True)
master.geometry("500x500")

def main_loop():
	global ds
	if ((learning==1) and (compute ==0)):
                print ("Inputs: ")
                print (tuple(data_input.values()))
                print ("Outputs: ")
                print (tuple( data_output.values()))
                ds.addSample(tuple(data_input.values()),tuple(data_output.values()))		
	if (l_map.poll(1)) and ((compute==1) and (learning==0)):
			#print "inputs to net: ", data_input
			activated_out=net.activate(tuple(data_input.values()))
			#print "Activated outs: ", activated_out
			for out_index in range(num_outputs):
				data_output[out_index]=activated_out[out_index]
				sliders[out_index].set(activated_out[out_index])
				l_outputs[out_index].update(data_output[out_index])


def on_gui_change(x,s_index):
#   s_index=0
	try:
		#print "in callback: on gui change"
		#print x,s_index
	
		global data_output    
		if (compute==0):
			data_output[s_index]=float(x)
			l_outputs[s_index].update(float(x))
			#print ("on gui change: ", data_output)
			#l_map.poll(0)

	except:
		print ("WTF MATE? On Gui Change Error!")
		raise

for s_index in range(num_outputs):
	def tc(s_index):
		return lambda x: on_gui_change(x,s_index)

	sliders[s_index]=Tkinter.Scale(master,from_=0,to=1, label='output'+str(s_index),orient=Tkinter.HORIZONTAL,length=300, resolution=0.01, command=tc(s_index))
	sliders[s_index].pack()



def learn_callback():
	global learning

	if learning == 1:
		b_learn.config(relief='raised',text="Acquire Training Data (OFF)",bg='gray')
		learning=0
		
		print ("learning is now OFF")
	elif learning ==0:
		b_learn.config(relief='sunken',text="Acquiring Training Data (ON)",bg='red')
		learning=1
		print ("learning is now ON")


	print ("learning is", learning)
	#b.learn_on.text="Acquire Training Data (ON)"

def compute_callback():
	global compute
	global net
	global ds
	if compute==1:
		b_compute.config(relief='raised',text="Press to compute network outputs (OFF)",bg='gray')
		compute =0
		print ("Compute network output is now OFF!")
	elif compute ==0:
		b_compute.config(relief='sunken',text="Computing network outputs(ON)",bg='coral')
		compute =1
		print ("Compute network output is now ON!")

def train_callback():
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

def clear_dataset():
	ds.clear()

def clear_network():
	#resets the module buffers but doesn't reinitialise the connection weights
	#TODO: reinitialise network here or make a new option for it.
	net.reset()
	
def save_dataset():
        save_filename = tkFileDialog.asksaveasfilename()
        ds.saveToFile(save_filename)
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
def load_dataset():
        open_filename = tkFileDialog.askopenfilename()
        global ds
        ds=SupervisedDataSet.loadFromFile(open_filename)
	print ds
        
def save_net():
        #from pybrain.tools.customxml import networkwriter
        save_filename = tkFileDialog.asksaveasfilename()
        networkwriter.NetworkWriter.writeToFile(net,save_filename)
def load_net():
        from pybrain.tools.customxml import networkreader
        open_filename = tkFileDialog.askopenfilename()
        global net
        net=networkreader.NetworkReader.readFrom(open_filename)

b_learn = Tkinter.Button(master, text="Acquire Training Data (OFF)", command=learn_callback)
b_learn.pack()	
b_train =Tkinter.Button(master, text="Train Network", command=train_callback)
b_train.pack()
b_compute = Tkinter.Button(master, text="Compute Network Outputs", command=compute_callback)
b_compute.pack()

b_clear_data=Tkinter.Button(master, text="Clear data set",command = clear_dataset)
b_clear_data.pack()
b_clear_net=Tkinter.Button(master, text="Reset Network",command = clear_network)
b_clear_net.pack()
b_save_dataset=Tkinter.Button(master, text='Save Current DataSet to file',command=save_dataset)
b_save_dataset.pack()
b_load_dataset=Tkinter.Button(master, text='Load DataSet from File',command=load_dataset)
b_load_dataset.pack()
b_save_net=Tkinter.Button(master, text='Save Current Network to File',command=save_net)
b_save_net.pack()
b_load_net=Tkinter.Button(master, text='Load Network from File',command=load_net)
b_load_net.pack()
                              

def ontimer():
	main_loop()
  #                  check the serial port
	master.after(10, ontimer)




#mapper signal handler (updates data_input[sig_indx]=new_float_value)
def h(sig, f):
	try:
		#print "mapper signal handler"
		#print (sig.name, f)

		global data_input
		global data_output
		
		#print sig.name
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

#create mapper signals (inputs)
for l_num in range(num_inputs):
	l_inputs[l_num]=l_map.add_input("/in%d"%l_num, 1, 'f',None,0,1.0, h)
	#l_map.poll(0)
	print ("creating input", "/in"+str(l_num))
	
#create mapper signals (outputs)
for l_num in range(num_outputs):
	l_outputs[l_num]=l_map.add_output("/out"+str(l_num), 1, 'f',None,0.0,1.0)
	l_inputs[l_num + num_inputs]=l_map.add_input("/out%d"%l_num, 1, 'f',None,0,1.0, h)
	#l_map.poll(0)
	print ("creating output","/out"+str(l_num))

#create network
net = buildNetwork(num_inputs,num_hidden,num_outputs,bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, recurrent=recurrent_flag)
#create dataSet
ds = SupervisedDataSet(num_inputs, num_outputs)
	
#while (True):
	
 

ontimer()
master.protocol("WM_DELETE_WINDOW", master.quit)
master.mainloop()
master.destroy()
del master
