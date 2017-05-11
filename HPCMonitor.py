import sys, subprocess, io, csv
from PyQt5 import QtWidgets
import mainwindow

def get_gpuinfo(query):
	output = subprocess.getoutput('nvidia-smi --query-gpu=%s --format=csv'%query)
	return [row for row in csv.reader(io.StringIO(output), delimiter=',')][1:]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        # self.ui.pushButton.clicked.connect(self.advanceSlider)
        gpuinfo = get_gpuinfo('name,compute_mode,utilization.gpu,utilization.memory')
        self.ui.label_GPU_name.setText(gpuinfo[0][0])

    # def advanceSlider(self):
        # self.ui.progressBar.setValue(self.ui.progressBar.value() + 1)

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
