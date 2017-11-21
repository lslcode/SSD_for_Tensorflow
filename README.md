# SSD_for_Tensorflow
Single Shot MultiBox Detector目标检测算法基于tensorflow的实现
写这套源码的目标是为了更好地学习SSD内部的原理，开源出来也为了大家有个学习的参考。
网上也有不少基于tensorflow实现ssd的源码，不过大多码得太复杂。
我也看了几套，然后就有一种强烈的冲动再码一套尽可能简单的源码，一方面可以更好地理解SSD的内部原理，另一方面也可以给各位初学者有个简单入门的源码参考。

与原论文不一致的地方：
1，box的位置信息论文描述为 [center_X, center_Y, width, height], 为了更好兼容和理解，这套源码统一改为[top_X, top_Y, width, height]

调用简单示例
1,检测
  ssd_model = ssd300.SSD300(tf_sess=sess, isTraining=False)
  f_class,f_location = ssd_model.run(input_img,None)
2,训练
  ssd_model = ssd300.SSD300(tf_sess=sess, isTraining=True)
  loss_all,loss_location,loss_class = ssd_model.run(train_data, actual_data)

【整体架码源码已完成，但是卷积参数还没有跟原论文一致】
