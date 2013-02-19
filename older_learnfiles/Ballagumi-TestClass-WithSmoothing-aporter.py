import mapper, sys, os, Tkinter, traceback, numpy
import matplotlib
matplotlib.use('TkAgg')

from multiprocessing import Process, Queue
from threading import Thread
from Queue import Empty
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

def main_loop():
    for b_num in open_list:
        try:
            b_array[b_num].get_serial_data()
            b_array[b_num].update_mapper_signals()
            #Smooth the data before it's sent out
            b_array[b_num].smooth_mapper_signal()
        except Exception, e:
            traceback.print_exc()
    try:
        m_inst.poll(0)
        queue_list=q.get_nowait()
        print ("queue list: ", queue_list)
        if queue_list=="quit":
            still_alive=0
            return
        else:
            if (queue_list[0] in open_list):
                print("push to com: ", queue_list[0])
                print(repr(queue_list[1]+"\r"))
                if queue_list[1].startswith('Sb'):
                    input_mode = queue_list
                elif queue_list[1].startswith('SI'):
                    get_board_number = 1
                b_array[queue_list[0]].s_write(queue_list[1]+"\r")
                print ("in waiting ", b_array[queue_list[0]].ser.inWaiting())
                ser_stream=(b_array[queue_list[0]].s_read(10))
                print ("ser stream: ", ser_stream)
                for ser_line in ser_stream:
                    print("Returned : ",ser_line)
            return
    except: 
        pass    

def Fung_GUI(q,on_list):
    import Tkinter
    def tcall(b_num):
        temp_str=tbox.get()
        q.put((b_num,temp_str))
        
    def onquit():
        q.put("quit")
        root.quit()
    
    print "in gui"
    root = Tkinter.Tk()
    
    def ontimer():
        main_loop()
        root.after(1, ontimer) #check the serial port

    
    def get_canvas():
        fig = Figure(figsize=(5,4), dpi=100)
        canvas = FigureCanvasTkAgg(fig,master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        return canvas
        
    def update_plot(canvas,data):
        print "UPDATED PLOT: Smooth Data", data
        
    def plottimer():
        smooth_data=[]
        for b_num in open_list:
            smooth_data.append(b_array[b_num].get_smoothed_signals())
        
        canvas = get_canvas()
        update_plot(canvas, smooth_data)
        root.after(100, plottimer) #update the plot every 100 ms

    #Text Box
    tbox=Tkinter.Entry(root)
    tbox.pack()
    
    #Quit Button
    quit_button = Tkinter.Button(root, text="Quit", command=lambda: onquit())
    quit_button.pack()
    
    #Board Buttons
    command_button={}
    for bd_num in on_list:
        txt_str=("Enter command for Board " + str(bd_num))
        def tc(b):
            return lambda: tcall(b)
        command_button[bd_num] = Tkinter.Button(root, text=txt_str,command=tc(bd_num))
        command_button[bd_num].pack()
        print ("command button" ,command_button)
    
    ontimer()
    plottimer()
    root.mainloop()
    
#Open Serial Port to Read Data
OS = os.name
if os.name == 'nt':
    port_list = [14-1,5-1,6-1]
elif os.name == 'posix':
    port_list=['/dev/tty.usbserial-A8004lV1','/dev/tty.usbserial-A8004lUY','/dev/tty.usbserial-A8004lUZ']
    
from fungible_board_class import Fungible_Node
print Fungible_Node

m_inst= mapper.device("Fungible1", 9000)
b_array={}
b_num=0
#b_list=[0,1,2]
b_list=[0]
open_list=[]
for b_num in b_list:
    try:
        b_array[b_num]=Fungible_Node(port_list[b_num],115200,0.3,m_inst)
        open_list.append(b_num)
    except:
        print ("error on b_num",b_num, port_list[b_num])
        raise

def cleanup():
    print ("auto-closing")
    for b_num in open_list:
        try:
            b_array[b_num].close_nicely()
        except:
            print ("board ", b_num, "already closed (or just plain couldn't close)!")

try:
    q = Queue()
    print q
except:
    print ("Queue couldn't open")

try:    
    Fung_GUI(q,open_list)
    cleanup()
except:
    print("GUI couldn't start")
    cleanup()
    raise


