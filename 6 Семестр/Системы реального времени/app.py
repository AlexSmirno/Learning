from PyQt6 import QtCore, QtGui, QtWidgets
import joblib
from graphviz import Source
from sqlalchemy import text
from sqlalchemy import create_engine
import sklearn
import numpy as np
import pandas as pd

engine = create_engine('postgresql+psycopg2://postgres:admin@localhost/real_time_system')

model_cr = joblib.load("models\model_FeCr.pkl")
model_mo = joblib.load("models\model_FeMo.pkl")
model_v = joblib.load("models\model_FeV.pkl")

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(735, 522)
        Dialog.setStyleSheet("background-color: rgb(163, 163, 182);")
        self.heat_number_label = QtWidgets.QLabel(parent=Dialog)
        self.heat_number_label.setGeometry(QtCore.QRect(80, 40, 211, 31))
        self.heat_number_label.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.heat_number_label.setObjectName("heat_number_label")
        self.heart_number_input = QtWidgets.QLineEdit(parent=Dialog)
        self.heart_number_input.setGeometry(QtCore.QRect(300, 30, 241, 51))
        self.heart_number_input.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(163, 163, 182);\n"
"selection-background-color: rgb(163, 163, 182);\n"
"alternate-background-color: rgb(163, 163, 182);\n"
"")
        
        self.heart_number_input.setObjectName("heart_number_input")
        self.fe_cr_lable = QtWidgets.QLabel(parent=Dialog)
        self.fe_cr_lable.setGeometry(QtCore.QRect(60, 380, 111, 31))
        self.fe_cr_lable.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.fe_cr_lable.setObjectName("fe_cr_lable")
        self.predict_button = QtWidgets.QPushButton(parent=Dialog)
        self.predict_button.setGeometry(QtCore.QRect(40, 230, 141, 61))
        self.predict_button.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(163, 163, 140);")
        self.predict_button.setObjectName("predict_button")
        self.fe_cr_lable_num = QtWidgets.QLabel(parent=Dialog)
        self.fe_cr_lable_num.setGeometry(QtCore.QRect(60, 440, 111, 31))
        self.fe_cr_lable_num.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.fe_cr_lable_num.setObjectName("fe_cr_lable_num")
        self.fe_mo_lable_num = QtWidgets.QLabel(parent=Dialog)
        self.fe_mo_lable_num.setGeometry(QtCore.QRect(290, 440, 111, 31))
        self.fe_mo_lable_num.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.fe_mo_lable_num.setObjectName("fe_mo_lable_num")
        self.fe_mo_lable = QtWidgets.QLabel(parent=Dialog)
        self.fe_mo_lable.setGeometry(QtCore.QRect(290, 380, 111, 31))
        self.fe_mo_lable.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.fe_mo_lable.setObjectName("fe_mo_lable")
        self.fe_v_lable_num = QtWidgets.QLabel(parent=Dialog)
        self.fe_v_lable_num.setGeometry(QtCore.QRect(540, 440, 111, 31))
        self.fe_v_lable_num.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.fe_v_lable_num.setObjectName("fe_v_lable_num")
        self.fe_v_lable = QtWidgets.QLabel(parent=Dialog)
        self.fe_v_lable.setGeometry(QtCore.QRect(540, 380, 111, 31))
        self.fe_v_lable.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";\n"
"border-color: rgb(255, 255, 255);")
        self.fe_v_lable.setObjectName("fe_v_lable")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.add_function()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.heat_number_label.setText(_translate("Dialog", "Номер плавки:"))
        self.fe_cr_lable.setText(_translate("Dialog", "FeCr"))
        self.predict_button.setText(_translate("Dialog", "Оценка"))
        self.fe_cr_lable_num.setText(_translate("Dialog", ""))
        self.fe_mo_lable_num.setText(_translate("Dialog", ""))
        self.fe_mo_lable.setText(_translate("Dialog", "FeMo"))
        self.fe_v_lable_num.setText(_translate("Dialog", ""))
        self.fe_v_lable.setText(_translate("Dialog", "FeV"))

    def add_function(self):
        self.predict_button.clicked.connect(lambda: self.write_number(self.heart_number_input.text()))

    def write_number(self, number):
        data = self.get_dataset(number)
        if data is None:
            return
        
        #data_cr = list(data.iloc[0][model_cr.feature_names_in_])
        #data_mo = list(data.iloc[0][model_mo.feature_names_in_])
        #data_v = list(data.iloc[0][model_v.feature_names_in_])
        
        data_cr = list(data.iloc[0][model_cr.feature_name_])
        data_mo = list(data.iloc[0][model_mo.feature_name_])
        data_v = list(data.iloc[0][model_v.feature_name_])

        res_cr = str(np.round((model_cr.predict([data_cr]))[0], 3))
        res_mo = str(np.round((model_mo.predict([data_mo]))[0], 3))
        res_v = str(np.round((model_v.predict([data_v]))[0], 3))

        self.fe_cr_lable_num.setText(res_cr)
        self.fe_mo_lable_num.setText(res_mo)
        self.fe_v_lable_num.setText(res_v)

    def get_dataset(self, number):
        heat_num = 0
        try:
             heat_num = int(number)
        except:
             print("Wrong input")
             return None
        
        try:
             with engine.begin() as conn:
                 data = pd.read_sql_query(text(f'SELECT * FROM final_table where \"HeatNo\" = {heat_num};'), conn, index_col="HeatNo")
             if data.empty or len(data) == 0:
                 print("There is not this heat")
                 return None
        except:
             print("There is not this heat")
             return None
        return data.fillna(0)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
