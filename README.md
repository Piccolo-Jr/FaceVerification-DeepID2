# FaceVerification-DeepID2
人脸验证卷积神经网络结构-DeepID2<br>

DeepID2人脸识别算法采用人脸识别信号和人脸验证信号同时训练卷积神经网络，不同于通常siamese network采用的triplet loss,DeepID2每次将两张人脸图像送入网络训练，人脸识别信号通过分类交叉熵损失来计算，人脸验证信号通过两张人脸的身份差异来创建L1/L2范数或余弦相似度的验证函数，本项目使用L2范数。<br>

（具体算法理解请参阅论文:Deep Learning Face Representation by Joint Identification-Verification）<br>

原实验采用200个卷积神经网络对200个人脸patches进行训练，每个神经网络生成2个160维DeepID2向量；再通过前向后向贪心算法筛选出总共25个有效且互补的DeepID2，形成25*160=4000维的DeepID2向量，最后通过PCA降到180维；用提取的DeepID2训练联合贝叶斯模型，通过两张人脸图像的联合概率来判断是否属于同一个人。<br>

本项目是对DeepID2提取特征部分神经网络的实现，使用tensorflow.keras构建网络结构，使用tensorflow.estimator训练模型，满足生产环境使用。<br>

项目结构如下<br>
```
├── data                          #存放原始人脸图像数据
│   ├── CelebFaces+
│   └── LFW
├── file                          #存放序列化数据
│   ├── CelebFaces+
│   └── LFW
├── models                        #存放模型文件
└── src                           #存放代码文件
    ├── DeepID2_model.py
    ├── face_data_generate.py
    └── face_data_prepare.py
```
网络在未添加验证信号，只用识别信号的条件下，在LFW数据集的正脸部分上训练，分类精度达到几乎100%，验证集也达到了97%。<br>
网络在使用识别信号加验证信号后扩大训练数据集，在CelebFaces+数据集上进行实验，单个神经网络的收敛情况不佳；在有算力条件支持的情况下，若完全按照原实验设置通过对200个神经网络训练并进行后续操作，理论上能达到99%的识别准确率。
