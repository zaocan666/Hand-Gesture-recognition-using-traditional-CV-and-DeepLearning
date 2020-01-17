# 手势数字0~5的识别
本项目用传统cv算法和卷积神经网络实现手势识别
## 可执行程序
可执行程序基于传统cv算法，打开ui.exe，选择摄像头，打开摄像头，在画面矩形框内影像稳定时点开始识别，将手放入矩形框画面内，下方显示识别结果。由于TensorFlow打包占用空间太大，所以没有基于卷积神经网络的可执行程序。
## 演示视频
演示视频包括传统方法的演示和神经网络的演示。
## 代码
### 传统cv方法
在 代码/traditional文件夹内 

- config.py: 存放一些配置变量的值 
- extract_hand_pic.py: 从静态图片中提取手部形状 
- extract_hand_video: 从视频流中提取手部形状 
- hand_number.py: 从提取出来的手部形状中识别出手势 
- main.py: 静态图片手势识别的主逻辑 
- ui.py: 用户界面程序 

### 基于卷积神经网络的方法
在 代码/deeplearning文件夹内，基于项目： https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network
#### 收集数据
执行PalmTracker.py文件收集训练和测试数据，数据保存在Dataset内（为减小体量，已删除数据）
#### 训练
执行ModelTrainer.ipynb进行训练，训练后的参数保存在TrainedModel文件夹内
#### 预测
执行ContinuousGesturePredictor.py进行预测，摄像头打开后等待1s，点s开始预测，然后将手放入画面矩形框内。
### 环境
- tensorflow 1.12.0
- python 3.5.2
- Cuda compilation tools, release 10.1, V10.1.105
- cudnn 6
### 训练参数
- **FC_LR = 1e-4** (learning rate of fully-connected parameters)
- **NET_LR = 1e-4** (learning rate of other parameters)
- **BATCH_SIZE = 64** (training batch size)
- **OPTIMIZER = 'adam'** (optimizer choice, adam or sgd)
- **WEIGHT_DECAY = 1e-4** (weight decay, applied only when using SGD)
- **MOMENTUM = 0.9** (momentum, applied only when using SGD)
- **DECAY_RATE = 0.1** (decay rate of learning rate every 10 epoches)
