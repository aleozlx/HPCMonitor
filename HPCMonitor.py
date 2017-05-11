import sys, subprocess, io, csv, re
from PyQt5 import QtCore, QtWidgets
import mainwindow

def get_gpuinfo(query):
    output = subprocess.getoutput('nvidia-smi --query-gpu=%s --format=csv'%query)
    return [row for row in csv.reader(io.StringIO(output), delimiter=',')][1:]

def percentage2int(s_percent):
    m = re.match(r'^\s*(\d+)\s*%\s*$', s_percent)
    return int(m.group(1)) if m else 0

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        
        timer_GPU = QtCore.QTimer(self)
        timer_GPU.timeout.connect(self.poll_GPU)
        timer_GPU.start(1000)
        self.poll_GPU()
        
    def poll_GPU(self):
        gpuinfo = get_gpuinfo('name,compute_mode,utilization.gpu,utilization.memory')
        print(gpuinfo)
        self.ui.label_GPU_name.setText(gpuinfo[0][0])
        self.ui.progressBar_GPU_util.setValue(percentage2int(gpuinfo[0][2]))
        self.ui.progressBar_GPU_mem.setValue(percentage2int(gpuinfo[0][3]))

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
