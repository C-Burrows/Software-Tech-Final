import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class CellPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Cellphone Price Prediction')
        #self.data = pd.read_csv('Cellphone.csv')
        self.sliders = []
        self.data_path = 'C:/CSP_CellData/archive/Cellphone.csv'  # Updated with your actual file path
        self.data = pd.read_csv(self.data_path)

        #remove columns not needed for the model training
        self.X = self.data.drop(['Price','Sale','Product_id'], axis=1).values

        self.y = self.data['Price'].values


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        for i, column in enumerate(self.data.columns[3:-1]):
            label = tk.Label(self.master, text=column + ': ')

            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i, column=2)
            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i, column=1)
            self.sliders.append((slider, current_val_label))

        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[3:-1]), columnspan=3)



    def predict_price(self):
        #grab the inputs from the screen and add in one for price we we don't want to show on the screen.
        inputs = [float(slider.get()) for slider, _ in self.sliders] +[0.0]
        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Price', f'The predicted Cellphone price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = CellPricePredictionApp(root)
    root.mainloop()
