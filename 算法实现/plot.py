import matplotlib.pyplot as plt

# 画图原点图
# 'ro' r 为颜色 o 为形状 
# markersize 大小
# zorder 绘画顺序
def test_validate(x_test, y_test, y_predict, classifier):
    x = range(len(y_test))
    plt.plot(x, y_test, "ro", markersize=5, zorder=3, label=u"true_v")
    plt.plot(x, y_predict, "go", markersize=8, zorder=2, label=u"predict_v,$R^2$=%.3f" % classifier.score(x_test, y_test))
    # 标签的位置
    plt.legend(loc="upper left")
    plt.xlabel("number")
    plt.ylabel("true?")
    plt.show()
