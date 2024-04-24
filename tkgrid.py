import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Treeview

plt.rcParams["font.sans-serif"] = "SimHei"  # 显示中文 黑体中文
plt.rcParams["axes.unicode_minus"] = False  # 显示负号

# 定义神经网络
class ELM:
    def __init__(self, hiddenNodeNum, activationFunc="sigmoid", type_="REGRESSION"):
        # beta矩阵
        self.beta = None
        # 偏置矩阵
        self.b = None
        # 权重矩阵
        self.W = None
        # 隐含层节点个数
        self.hiddenNodeNum = hiddenNodeNum
        # 激活函数
        self.activationFunc = self.chooseActivationFunc(activationFunc)
        # 极限学习机类别   :CLASSIFIER->分类， REGRESSION->回归
        self.type_ = type_

    def fit(self, X, T):
        if self.type_ == "REGRESSION":
            try:
                if T.shape[1] > 1:
                    raise ValueError("回归问题的输出维度必须为1")
            except IndexError:
                # 如果数据是一个array，则转换为列向量
                T = np.array(T).reshape(-1, 1)
        if self.type_ == "CLASSIFIER":
            # 独热编码器
            encoder = OneHotEncoder()
            # 将输入的T转换为独热编码的形式
            T = encoder.fit_transform(T.reshape(-1, 1)).toarray()
        # 输入维度d，输出维度m，样本个数N，隐含层节点个数hiddenNodeNum
        n, d = X.shape
        # 权重系数矩阵 d*hiddenNodeNum
        self.W = np.random.uniform(-1.0, 1.0, size=(d, self.hiddenNodeNum))
        # 偏置系数矩阵 n*hiddenNodeNum
        self.b = np.random.uniform(-0.4, 0.4, size=(n, self.hiddenNodeNum))
        # 隐含层输出矩阵 n*hiddenNodeNum
        H = self.activationFunc(np.dot(X, self.W) + self.b)
        # 输出权重系数 hiddenNodeNum*m，β的计算公式为：((H.T*H)^-1)*H.T*T
        self.beta = np.dot(np.dot(pinv(np.dot(H.T, H)), H.T), T)

    def chooseActivationFunc(self, activationFunc):
        """选择激活函数，这里返回的值是函数名"""
        if activationFunc == "sigmoid":
            return self._sigmoid
        elif activationFunc == "sin":
            return self._sine
        elif activationFunc == "cos":
            return self._cos

    def predict(self, x):
        # 样本个数
        sampleCNT = len(x)
        # 由于训练样本个数为b矩阵的行，用len函数获取，进行预测的时候必须满足该条件，否则下面公式索引会超出范围
        if sampleCNT > len(self.b):
            raise ValueError("训练集样本数必须大于测试机样本数")
        h = self.activationFunc(np.dot(x, self.W) + self.b[:sampleCNT, :])
        res = np.dot(h, self.beta)
        if self.type_ == "REGRESSION":  # 回归预测
            return res
        elif self.type_ == "CLASSIFIER":  # 分类预测
            # 返回最大值所在位置的索引，因为最大值位置的类别恰好等于索引
            return np.argmax(res, axis=1)
    @staticmethod
    def score(y_true, y_pred):
        # 根据输出标签相等的个数计算得分
        if len(y_pred) != len(y_true):
            raise ValueError("维度不相等")
        totalNum = len(y_pred)
        rightNum = np.sum([1 if p == t else 0 for p, t in zip(y_pred, y_true)])
        return rightNum / totalNum

    @staticmethod
    def RMSE(y_pred, y_true):
        # Root Mean Square Error    均方根误差
        # 这里计算平均均方根误差
        # 计算公式参考：https://blog.csdn.net/yql_617540298/article/details/104212354
        try:
            if y_pred.shape[1] == 1:
                y_pred = y_pred.reshape(-1)
        except IndexError:
            pass

        return np.sqrt(np.sum(np.square(y_pred - y_true)) / len(y_pred))
    def MAE(self, y_pred, y_true):
        # Root Mean Square Error    平均绝对误差
        try:
            if y_pred.shape[1] == 1:
                y_pred = y_pred.reshape(-1)
        except IndexError:
            pass

        return np.sum(np.abs(y_true-y_pred))/len(y_true)
    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    @staticmethod
    def _sine(x):
        return np.sin(x)

    @staticmethod
    def _cos(x):
        return np.cos(x)






# 加载数据
def import_data():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])

    if file_path:
        try:
            global df
            df = pd.read_excel(file_path)
            data_table.delete(*data_table.get_children())  # 清空表格
            for index, row in df.iterrows():
                data_table.insert("", "end", values=list(row))
            messagebox.showinfo("Success", "Data imported successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

        # 定义数据标准化函数

def standardize_data():
    if 'df' not in globals():
        messagebox.showerror("Error", "Please import data first.")
        return
    global X, y
    X = df.iloc[1:, :-1]  # 前七列作为输入特征
    y = df.iloc[1:, -1]  # 最后一列作为输出目标
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    # 重置 y 的索引为从 0 开始.
    y_reset = y.reset_index(drop=True)
    # 将重置索引后的 y 添加到 df_scaled 中作为 'Target' 列
    df_scaled['Target'] = y_reset
    global df1
    df1 = df_scaled
    update_table()
    messagebox.showinfo("Success", "Data standardized successfully.")


# 定义划分数据函数
def split_data():
    if 'df1' not in globals():
        messagebox.showerror("Error", "Please import and standardize data first.")
        return
    global X_train, X_test, y_train, y_test
    X = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    messagebox.showinfo("Success", "Data split successfully.")

# 更新表格数据
def update_table():
    data_table.delete(*data_table.get_children())  # 清空表格
    for index, row in df1.iterrows():
        data_table.insert("", "end", values=list(row))



def crate_ELM():
    global elm
    # global input_nodes
    global hiddenNodeNum
    # global output_nodes
    global activationFunc
    global type_
    # —————————————创建神经网络对象并用数据集训练网络——————————#
    # 输入、隐藏、输出节点数

    hiddenNodeNum = int(var_hidden.get())
    activationFunc = str(var_activationFunc.get())
    type_ = str(var_type_.get())
    # 创建神经网络对象
    elm = ELM(hiddenNodeNum, activationFunc, type_)
    text.insert(tk.END, 'elm网络构建成功！\n')
    text.insert(tk.END,
                 '隐藏层节点数：' + var_hidden.get() + '\n')
    text.insert(tk.END, '激活函数：' + var_activationFunc.get() + '，类型：' + var_type_.get() + '\n')
    text.insert(tk.END, '可以开始训练了！\n')


# 开始训练函数，训练数据集
def beg_train():

    try:
        # 假设你已经有了一个初始化好的网络和数据
        elm.fit(X_train, y_train)
        text.insert(tk.END, '训练完毕！\n')
        text.insert(tk.END, '可以开始测试你的网络了!\n')
    except Exception as e:
        text.insert(tk.END, f"训练过程中发生错误: {e}\n")
        messagebox.showerror("Error", str(e))
pass


#       打开测试数据集MNIST-test

# 开始测试函数，遍历所有测试集中的测试数据，得出准确率

# ————————————————————————测试MNIst数据集————————————————————#
def beg_test():
    try:
        global y_pred
        text.insert(tk.END, '开始预测青霉素浓度.....\n')
        # 调用预测方法
        y_pred = elm.predict(X_test)

        # 显示预测结果
        result_text = "预测结果:\n"
        if elm.type_ == "REGRESSION":
            result_text += "\n".join(map(str, y_pred))
        elif elm.type_ == "CLASSIFIER":
            result_text += "\n".join(["类别: " + str(pred) for pred in y_pred])
        result_widget.delete("1.0", tk.END)
        result_widget.insert(tk.END, result_text)
        # 显示预测结果对比弹窗
        show_prediction_comparison(y_pred, y_test)
    except Exception as e:
        messagebox.showerror("错误", str(e))
    rmse = elm.RMSE(y_pred, y_test)
    print("平均RMSE为：", rmse)
    mae = elm.MAE(y_pred, y_test)
    print("平均绝对误差为：", mae)
    # # 评估模型——只适用于分类
    # accuracy = elm.score(y_test, y_pred)
    # print(f"Model accuracy: {accuracy:.2f}")
    # n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    # # 用来存放分数，即正确率
    # scorecard = []

    # for record in X_test:
    #     # 用”，“号分开数据
    #     all_values = record.split(',')
    #     # 用准确值标签记录数字准确值
    #     correct_label = int(all_values[0])
    #     print("---------")
    #     print("正确结果", correct_label)
    #     # 缩放
    #     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #     # 计算输出
    #     outputs = n.query(inputs)
    #     # 输出的最大值即为判断值
    #     label = np.argmax(outputs)
    #     print("神经网络判断", label)
    #     # 将正确和错误的判断形成一个列表
    #     if (label == correct_label):
    #         # 正确为1
    #         scorecard
    #         scorecard.append(1)
    #     else:
    #         # 错误为0
    #         scorecard
    #         scorecard.append(0)
    # print(scorecard)
    # scorecard_array = np.asarray(scorecard)
    # # 正确率
    # right_rate = (scorecard_array.sum() / scorecard_array.size) * 100
    #
    text.insert(tk.END, '数据测试完毕\n')
    text.insert(tk.END, '均方误差： ' + str(rmse) + '\n')
    text.insert(tk.END, '平均绝对误差为： ' + str(mae) + '\n')
    text.update()
    pass


def show_prediction_comparison(y_pred, y_test):
    global result_window
    # 创建一个新的Tkinter弹窗
    result_window = tk.Toplevel(window)
    result_window.title('测试集预测结果对比')

    # 创建一个matplotlib的Figure对象
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(y_pred)), y_pred, 'b-o', linewidth=1, label='预测值')
    ax.plot(range(len(y_test)), y_test, 'r-*', linewidth=1, label='真实值')
    ax.legend()
    ax.set_xlabel('预测样本', size=14)
    ax.set_ylabel('预测结果', size=14)
    ax.set_title('测试集预测结果对比')

    # 将matplotlib的Figure对象转换为Tkinter可以嵌入的部件
    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()

    # 将画布部件添加到弹窗中
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # 运行Tkinter的事件循环，确保弹窗保持打开状态
    result_window.protocol("WM_DELETE_WINDOW", on_closing)  # 绑定关闭事件，避免直接关闭导致程序崩溃
    result_window.mainloop()

def on_closing():
    # 当关闭弹窗时，隐藏而不是销毁它，以避免整个程序的Tkinter事件循环结束
    result_window.withdraw()



# img_frame = tk.LabelFrame(window, text='图像显示', padx=10, pady=10,
#                           width=120, height=120)
# img_frame.place(x=55, y=50)
# ——————————————初始化GUI界面——————————————--#
window = tk.Tk()
window.title('ELM预测青霉素浓度')
window.geometry('720x540')
global img_png  # 定义全局变量 图像的
var = tk.StringVar()  # 这时文字变量储存器
text = tk.Text(window, width=20, height=15)
text.grid(row=5, column=0, columnspan=6,
          padx=10, pady=10, sticky="nesw")
text.insert(tk.END, '请输入相关数据，构建一个网络\n')

# 创建数据表格
columns = ["Column " + str(i) for i in range(1, 9)]  # 假设最多8列数据
data_table = Treeview(window, show="headings", columns=columns)
for col in columns:
    data_table.heading(col, text=col)
    data_table.column(col, width=30)
data_table.grid(row=0, column=0, columnspan=4, rowspan=4,
                padx=10, pady=10, sticky="nesw")

window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=1)
window.columnconfigure(3, weight=1)
window.columnconfigure(4, weight=1)
window.columnconfigure(5, weight=1)
window.rowconfigure(0, weight=1)
window.rowconfigure(3, weight=1)
# window.rowconfigure(3, weight=1)
# window.rowconfigure(4, weight=1)
window.rowconfigure(5, weight=1)
# frame2.columnconfigure(0, weight=1)
# frame3.rowconfigure(1, weight=1)

# 创建显示结果的文本框
result_widget = scrolledtext.ScrolledText(window, height=25, width=25)
result_widget.grid(row=3, column=4, columnspan=2,
                   padx=10, pady=10, sticky="nesw")

var_frame1 = tk.Frame(window)
var_frame1.grid(row=0, column=4, columnspan=2,
                   sticky="nesw")
# var_frame1.rowconfigure(1, weight=1)
# var_frame1.columnconfigure(0, weight=1)

var_frame2 = tk.Frame(window)
var_frame2.grid(row=1, column=4, columnspan=2,
                   sticky="nesw")
var_frame2.rowconfigure(1, weight=1)
var_frame3 = tk.Frame(window)
var_frame3.grid(row=2, column=4, columnspan=2,
                   sticky="nesw")
var_frame3.rowconfigure(1, weight=1)

# 创建文本窗口，显示当前操作8状态
hi_lable = tk.Label(var_frame1, text='隐藏层节点数：')
# hi_lable.grid(row=0, column=4, sticky="w")
hi_lable.pack(pady=5, side="left")
var_hidden = tk.StringVar()
var_hidden.set('50')
entry_hidden = tk.Entry(var_frame1, textvariable=var_hidden, width=12)
# entry_hidden.grid(row=0, column=5, sticky="w")
entry_hidden.pack(pady=5, side="left")

activationFunc_lable = tk.Label(var_frame2, text='激活函数：')
# activationFunc_lable.grid(row=1, column=4, ipady=2, sticky="w")
activationFunc_lable.pack(pady=5, side="left")

var_activationFunc = tk.StringVar()
var_activationFunc.set('sigmoid')
entry_activationFunc = tk.Entry(var_frame2, textvariable=var_activationFunc, width=12)
# entry_activationFunc.grid(row=1, column=5, ipady=2, sticky="w")
entry_activationFunc.pack(padx=23, pady=5, side="left")

type__lable = tk.Label(var_frame3, text='类型：')
# type__lable.grid(row=2, column=4, sticky="w")
type__lable.pack(pady=5, side="left")

var_type_ = tk.StringVar()
var_type_.set('REGRESSION')
entry_type_ = tk.Entry(var_frame3, textvariable=var_type_, width=12)
# entry_type_.grid(row=2, column=5, ipady=2, sticky="w")
entry_type_.pack(padx=47, pady=5, side="left")


# 导入数据按钮
btn_import = tk.Button(window, text='导入数据', width=12, height=2,
                       command=import_data)
btn_import.grid(row=4, column=0, padx=10, sticky="ew")

# 数据标准化按钮
btn_std = tk.Button(window, text='标准化数据', width=12, height=2,
                      command=standardize_data)
btn_std.grid(row=4, column=1, padx=10, sticky="ew")

# 划分数据集按钮
btn_split = tk.Button(window, text='划分数据集', width=12, height=2,
                      command=split_data)
btn_split.grid(row=4, column=2, padx=10, sticky="ew")

# 神经网络初始化按钮
btn_train = tk.Button(window, text='构建网络', width=12, height=2,
                      command=crate_ELM)
btn_train.grid(row=4, column=3, padx=10, sticky="ew")

# 训练数据集按钮
btn_test = tk.Button(window, text='训练数据集', width=12, height=2,
                     command=beg_train)
btn_test.grid(row=4, column=4, padx=10, sticky="ew")


# 测试数据集按钮
btn_predict = tk.Button(window, text='测试数据集', width=12, height=2,
                     command=beg_test)  # 点击按钮式执行的命令
btn_predict.grid(row=4, column=5, padx=10, sticky="ew")



# 创建显示图像按钮
# btn_Show = tk.Button(window,
#                      text='打开测试图片',  # 显示在按钮上的文字
#                      width=15, height=2,
#                      command=Open_Img)  # 点击按钮式执行的命令
#
# btn_Show.pack()
# # 按钮位置
# btn_Show.place(x=450, y=210)
# 运行整体窗口
window.mainloop()
pass





plt.figure(1)
plt.title('测试集预测结果对比')
plt.plot(range(len(y_pred)), y_pred, 'b-o', linewidth=1, label='预测值')
plt.plot(range(len(y_test)), y_test, 'r-*', linewidth=1, label='真实值')
plt.legend()
plt.xlabel('预测样本', size=14)
plt.ylabel('预测结果', size=14)
plt.show()


