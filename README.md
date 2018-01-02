# SSD_for_Tensorflow
  
Single Shot MultiBox Detector目标检测算法基于tensorflow的实现<br/>
论文在<a href='https://arxiv.org/abs/1512.02325' target='_blank'>这里</a>
<br/><br/>
网上也有不少基于tensorflow实现ssd的源码，不过大多码得太复杂。<br/>
我看了几套，然后就有一种强烈的冲动再码一套尽可能简单直接的源码，一方面可以更好地理解SSD的内部原理，另一方面也可以给各位初学者有个简单入门的源码参考。<br/>
<br/>
代码结构：<br/>
<b style='color:#ff5500'>ssd300.py</b> - ssd的核心代码封装，实现300 * 300的图片格式。<br/>
<b style='color:#ff5500'>main.py</b> - ssd300的使用用例，包括训练、检测的调用示例。训练时使用voc2012数据集，数据可以从<a href='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' target='_blank'>这里</a>下载，解压到\train_datasets\voc2012目录下
<br/>
<br/>
因为要简洁，就只有这2个文件。
<br/>
<br/>
与原论文不一致的地方：<br/>
<span style='text-decoration:line-through'><b>1</b>，box的位置信息论文描述为 [center_X, center_Y, width, height], 为了更好兼容和理解，这套源码统一改为[top_X, top_Y, width, height]</span><br/>
<br/><br/>
<b>2</b>，论文中default box的width=scale*sqrt(aspect_ratio)、height=scale/sqrt(aspect_ratio) 是错误的，<br/>改为width=sqrt(scale * aspect_ratio)、height=sqrt(scale/aspect_ratio)，有兴趣的朋友可以反推一下。<br/><br/>
<b>3</b>，按照论文中描述长宽比ratio = 1时，scale=sqrt(scale0 * scale1)，即值为sqrt(1.0 * 2.0)=1.414，与scale4=1.5接近，不利于区分default box，因此直接修改为(scale0+scale4)/2=(1.0+1.5)/2=1.25，即取scale0和scale4中间值。<br/><br/>
<b>4</b>，论文中default_box_scale由公式s_k=s_min+(s_max-s_min) * (k-1)/(m-1)生成,源码改为np.linspace生成等差数组,效果一致<br/><br/>
<b>5</b>，box scale 由[ 0.2 , 0.9 ]改为[ 0.1 , 0.9 ]，因为最小box面积0.2不利于识别面积小的物体，所以改为0.1<br/><br/>


<br/><br/>
调用简单示例<br/>
1,检测<br/>
&nbsp;&nbsp;&nbsp;&nbsp;ssd_model = ssd300.SSD300(tf_sess=sess, isTraining=False)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;f_class,f_location = ssd_model.run(input_img,None)<br/>
2,训练<br/>
&nbsp;&nbsp;&nbsp;&nbsp;ssd_model = ssd300.SSD300(tf_sess=sess, isTraining=True)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;loss_all,loss_location,loss_class = ssd_model.run(train_data, actual_data)<br/>
<br/>
【整体框架源码已完成，可以参考学习。还没完成训练,可能还存在一些问题,如果发现有问题,请告诉我 : jasonli8848@qq.com】<br/>
<br/>
【注】<br/>
1，【经实验top_x,top_y并不适合卷积，会降低精度，应改为center_x,center_y】；<br/>
2，【源码中vgg基础网络并不完善，最好改为ResNet + Inception2】；<br/>
3，【default box 应根据具体业务设置，以免造成资源浪费以及影响精度】；<br/>

