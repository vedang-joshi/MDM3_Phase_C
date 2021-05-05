import matplotlib.pyplot as plt

m_scales = [300,200,100,50,25,20]
testscores = [0.8383838534355164,  0.8211920261383057, 0.813725471496582, 0.8042762875556946, 0.831285834312439, 0.8070784211158752]


labels = [300,200, 100, 50, 25,20]

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(m_scales, labels, rotation='horizontal')
plt.plot(m_scales, testscores)
plt.title('Varying data chunks used in sliding window')
plt.ylabel('Accuracy of test scores')
plt.xlabel('Size of data in chunks')
plt.show()

#[0.9950000047683716], [0.03965083882212639, 0.9934210777282715], [0.14299678802490234, 0.9592834115028381], [0.2818697690963745, 0.8993839621543884], [0.2967117130756378, 0.8953726291656494], [0.3853455185890198, 0.8516912460327148]]
