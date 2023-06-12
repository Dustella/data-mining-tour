import tkinter as tk

# 创建窗口
window = tk.Tk()
window.title("文本输入示例")

# 创建文本输入框
text_input = tk.Entry(window, width=500, show=None, font=("Arial", 14), bg="white", fg="black", bd= 5)
text_input.pack()

# 创建事件处理函数
def show_text():
    text = text_input.get()
    text_label.config(text="输入的文本是：" + text)

# 创建按钮
button = tk.Button(window, text="显示文本", font=("Microsoft YaHei",16),command=show_text)
button.pack()

# 创建文本标签
text_label = tk.Label(window, width=500, show=None, font=("Arial", 14), bg="white", fg="black", bd= 5, text="")
text_label.pack()

# 显示窗口
window.mainloop()