import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from matplotlib import pyplot as plt
import pandas as pd # pip install pandas
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.window_width, self.window_height = 800, 500
        self.resize(self.window_width, self.window_height)
        self.setWindowTitle('Forecast App')

        # creating a combo box widget
        self.combo_box = QComboBox(self)
 
        # setting geometry of combo box
        self.combo_box.setGeometry(200, 20, 300, 30)
 
        # geek list
        geek_list = ["Debriyaj Alt Merkezi Actros Y.M (Pentosinli)", "Debriyaj Seti 430 Mm Travego, Setra, Tourismo (Baskı, Balata, Bilya)", "Debriyaj Üst Merkezi Man 32.270- 19.463 / Temsa Safir E6", "Marş Dinamosu 457 Motor Euro 3, Euro 4, Euro 5", "Mazot Filtresi 1/2 Kg Keçe","Mazot Filtresi 457, 501, 502, Travego, Setra, 560 İntro", "Piston Kolu 502 La", "Supap Keçesi Axor 457, Actros, Travego - Daf Mx11 Eu6 (Yeni Versiyon)", "Yağ Filtresi Actros 501, 502, Travego, Setra Dingilli"]
 
        # adding list of items to combo box
        self.combo_box.addItems(geek_list)
 
        
        layout = QGridLayout()

        #layout.addWidget(QPushButton("Show content ", self), 0, 0)
        layout.addWidget(self.combo_box, 0, 0)
        self.table = QTableWidget()
        self.table.show()
        layout.addWidget(self.table,1,0)
        
        self.labelImage = QLabel(self)
        layout.addWidget(self.labelImage,1,1)
        
        self.labelImage2 = QLabel(self)
        layout.addWidget(self.labelImage2,3,1)
        
        self.setLayout(layout)

        self.button = QPushButton('&Load Data')
        self.button.clicked.connect(lambda _, xl_path=excel_file_path, sheet_name=worksheet_name: self.loadExcelData(xl_path, sheet_name))
        layout.addWidget(self.button,0,1)
        
        self.table2 = QTableWidget()
        self.table2.show()
        layout.addWidget(self.table2,3,0)
        
        
        self.table3 = QTableWidget()
        self.table3.show()
        layout.addWidget(self.table3,2,0)
        
        self.button2 = QPushButton('&Load Order Data')
        self.button2.clicked.connect(lambda _, xl_path=excel_file_path2, sheet_name=worksheet_name2: self.loadSaleData(xl_path, sheet_name))
        layout.addWidget(self.button2,2,1)
        
    def find(self):
 
        # finding the content of current item in combo box
        content = self.combo_box.currentText()
 
        # showing content on the screen through label
        self.label.setText("Content : " + content)
         
    def loadExcelData(self, excel_file_dir, worksheet_name):
        df = pd.read_excel(excel_file_dir, worksheet_name)
        if df.size == 0:
            return

        content = self.combo_box.currentText()
        
        df = df[df['Stock Name: '] == str(content)] #datasette anlatılacak ürün
        
        columns_to_drop = ['Order_Number','Invoice_Date', 'Current_Code', 'Stock_Code','Gross_Price_on_Order_Date','Quantity_Actual_Euro', 'Quantity_Actual_Tl']
        df = df.drop(columns=columns_to_drop)
        
        #df['Order_Date'] = df['Order_Date'].dt.to_period('M')
        df['Order_Date'] = pd.to_datetime(df['Order_Date']).dt.to_period('M')
        df['New _Order_Date']=df['New _Order_Date'].dt.to_period('M')
        
        df.dropna()
        
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)

        # returns pandas array object
        for row in df.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0,.0f}'.format(value)
                tableItem = QTableWidgetItem(str(value))
                self.table.setItem(row[0], col_index, tableItem)

        self.table.setColumnWidth(2, 300)
        
        monthly_sales = df.groupby('Order_Date').sum().reset_index()
        monthly_sales['Order_Date'] = monthly_sales['Order_Date'].dt.to_timestamp()
        monthly_sales['sales_diff'] = monthly_sales['Quantity_Actual'].diff()
        monthly_sales = monthly_sales.dropna()
        
        supverised_data = monthly_sales.drop(['Order_Date','Quantity_Actual'], axis=1)
        
        for i in range(1,13):
            col_name = 'month_' + str(i)
            supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
        supverised_data = supverised_data.dropna().reset_index(drop=True)
        
        train_data = supverised_data[:-6]
        test_data = supverised_data[-6:]
        
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        
        X_train, y_train = train_data[:,1:], train_data[:,0:1]
        X_test, y_test = test_data[:,1:], test_data[:,0:1]
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
        sales_dates = monthly_sales['Order_Date'][-6:].reset_index(drop=True)
        predict_df = pd.DataFrame(sales_dates)
        
        act_sales = monthly_sales['Quantity_Actual'][-7:].to_list()
        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False)) # 50 nöron sayısı
        model.add(Dense(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        history = model.fit(X_train, y_train, batch_size=1, epochs=200, validation_data=(X_test, y_test))
        
        metrics_df = pd.DataFrame(history.history)
        
        lstm_pred = model.predict(X_test, batch_size=1)
        lstm_pred = lstm_pred.reshape(-1,1)
        lstm_pred_test_set = np.concatenate([lstm_pred,X_test], axis=1)
        lstm_pred_test_set = scaler.inverse_transform(lstm_pred_test_set)
        result_list = []
        for index in range(0, len(lstm_pred_test_set)):
            result_list.append(lstm_pred_test_set[index][0] + act_sales[index])
        lstm_pred_series = pd.Series(result_list, name='lstm_pred')
        predict_df = predict_df.merge(lstm_pred_series, left_index=True, right_index=True)
        
        print(predict_df)
        
        self.table3.setRowCount(predict_df.shape[0])
        self.table3.setColumnCount(predict_df.shape[1])
        self.table3.setHorizontalHeaderLabels(predict_df.columns)

        # returns pandas array object
        for row in predict_df.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0,.0f}'.format(value)
                tableItem = QTableWidgetItem(str(value))
                self.table3.setItem(row[0], col_index, tableItem)

        self.table3.setColumnWidth(5, 300)
        
        
        pixmap = QPixmap("LSTM.PNG")
        self.labelImage.setPixmap(pixmap)
        
        
    
    def loadSaleData(self, excel_file_dir2, worksheet_name2):
        df = pd.read_excel(excel_file_dir2, worksheet_name2)
        if df.size == 0:
            return

        content = self.combo_box.currentText()
        
        df = df[df['Stock_Name'] == str(content)] #datasette anlatılacak ürün
        
        columns_to_drop = ['SIPARIS_NO','YIL', 'AY', 'HAFTA','FATURA_TARIHI','TESLIM_MIKTAR', 'FARK_MIKTAR','ST_GRUP_ACIKLAMA','ST_KOD1_ACIKLAMA','ST_KOD6_ACIKLAMA','ST_KOD7_ACIKLAMA','ST_KOD8_ACIKLAMA','ST_KOD9_ACIKLAMA',]
        df = df.drop(columns=columns_to_drop)
        
        #df['Order_Date'] = df['Order_Date'].dt.to_period('M')
        df['New_Order_Date'] = pd.to_datetime(df['New_Order_Date']).dt.to_period('M')
        
        df.dropna()
        
        self.table2.setRowCount(df.shape[0])
        self.table2.setColumnCount(df.shape[1])
        self.table2.setHorizontalHeaderLabels(df.columns)

        # returns pandas array object
        for row in df.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0,.0f}'.format(value)
                tableItem = QTableWidgetItem(str(value))
                self.table2.setItem(row[0], col_index, tableItem)

        self.table2.setColumnWidth(2, 300)
        
        pixmap2 = QPixmap("Sale.PNG")
        self.labelImage2.setPixmap(pixmap2)
        
        
        
if __name__ == '__main__':
    
    excel_file_path = 'dataset_updated.xlsx'
    worksheet_name = 'Result Data'
    
    excel_file_path2 = 'SaleData.xlsx'
    worksheet_name2 = 'Order Result'

    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 17px;
        }
    ''')
    
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')



