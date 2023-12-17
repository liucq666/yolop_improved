import os
from yacs.config import CfgNode as CN


_C = CN() #定义配置文件的结构
_C.LOG_DIR = 'runs/' #设置日志文件夹路径
_C.GPUS = (0, 1) #设置使用的GPU编号
_C.WORKERS = 0 #设置dataloader中使用的线程数
_C.PIN_MEMORY = False #是否将数据加载到内存中加速训练
_C.PRINT_FREQ = 2 #0打印信息的频率
_C.AUTO_RESUME = False #是否自动从断点处恢复训练
_C.NEED_AUTOANCHOR = False #是否需要重新选择先前的锚点(k-means)，在从头开始训练时(epoch=0)设为True!
_C.DEBUG = False #是否开启调试模式

_C.num_seg_class = 2 #分割类别的数量

_C.CUDNN = CN() #Cudnn相关参数
_C.CUDNN.BENCHMARK = True #是否对cudnn进行优化
_C.CUDNN.DETERMINISTIC = False #是否使用确定性算法进行计算
_C.CUDNN.ENABLED = True #是否启用Cudnn

_C.MODEL = CN(new_allowed=True) #神经网络信息相关参数
_C.MODEL.NAME = '' #网络名称
_C.MODEL.STRU_WITHSHARE = False #是否添加share_block到segbranch
_C.MODEL.HEADS_NAME = [''] #heads的名称列表
_C.MODEL.PRETRAINED = "" #预训练模型路径
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [640, 640] #图像大小
_C.MODEL.EXTRA = CN(new_allowed=True) #网络架构

_C.LOSS = CN(new_allowed=True) #损失函数相关参数
_C.LOSS.LOSS_NAME = '' #损失函数名称
_C.LOSS.MULTI_HEAD_LAMBDA = None #多头损失系数
_C.LOSS.FL_GAMMA = 2.0
_C.LOSS.CLS_POS_WEIGHT = 1.0 #分类损失正样本权重
_C.LOSS.OBJ_POS_WEIGHT = 1.0 #目标损失正样本权重
_C.LOSS.SEG_POS_WEIGHT = 1.0 #分割损失正样本权重
_C.LOSS.BOX_GAIN = 0.05 #box loss增益
_C.LOSS.CLS_GAIN = 0.5 #分类loss增益
_C.LOSS.OBJ_GAIN = 1.0 #目标loss增益
_C.LOSS.DA_SEG_GAIN = 0.3
_C.LOSS.LL_SEG_GAIN = 0.5
_C.LOSS.LL_IOU_GAIN = 0.2

_C.DATASET = CN(new_allowed=True) #数据集相关参数
_C.DATASET.DATAROOT = r'D:\yisa\YOLOP-main\data\images' #图片路径
_C.DATASET.LABELROOT = r'D:\yisa\YOLOP-main\data\100k' #标注文件夹路径
_C.DATASET.MASKROOT = r'D:\yisa\YOLOP-main\data\bdd_seg_gt' #驾驶区域分割标注文件夹路径
_C.DATASET.LANEROOT = r'D:\yisa\YOLOP-main\data\bdd_lane_gt' #车道线分割标注文件夹路径
_C.DATASET.DATASET = 'BddDataset' #数据集的名称
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg' #数据格式
_C.DATASET.SELECT_DATA = False #是否选择数据
_C.DATASET.ORG_IMG_SIZE = [720, 1280] #原始图像大小

#训练数据增强相关参数
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015
_C.DATASET.HSV_S = 0.7
_C.DATASET.HSV_V = 0.4

_C.TRAIN = CN(new_allowed=True) #训练相关参数
_C.TRAIN.LR0 = 0.001 #初始学习率
_C.TRAIN.LRF = 0.2 #最终OneCycleLR学习率
_C.TRAIN.WARMUP_EPOCHS = 3.0 #热身周期数
_C.TRAIN.WARMUP_BIASE_LR = 0.1 #热身期偏置学习率
_C.TRAIN.WARMUP_MOMENTUM = 0.8 #热身期动量
_C.TRAIN.OPTIMIZER = 'adam' #优化器
_C.TRAIN.MOMENTUM = 0.937 #动量
_C.TRAIN.WD = 0.005 #权重衰减
_C.TRAIN.NESTEROV = True #是否使用Nesterov加速梯度下降
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.BEGIN_EPOCH = 0 #开始训练的epoch
_C.TRAIN.END_EPOCH = 300 #结束训练的epoch
_C.TRAIN.VAL_FREQ = 1 #验证的频率
_C.TRAIN.BATCH_SIZE_PER_GPU = 32 #单个GPU上的batch size
_C.TRAIN.SHUFFLE = True #是否打乱数据
_C.TRAIN.IOU_THRESHOLD = 0.2 #iou阈值
_C.TRAIN.ANCHOR_THRESHOLD = 4.0 #锚点阈值

_C.TRAIN.SEG_ONLY = False #是否仅训练两个分割分支
_C.TRAIN.DET_ONLY = False #是否仅训练检测分支
_C.TRAIN.ENC_SEG_ONLY = False #是否仅训练编码器和两个分割分支
_C.TRAIN.ENC_DET_ONLY = False #是否仅训练编码器和检测分支
_C.TRAIN.DRIVABLE_ONLY = False #是否仅训练驾驶区域分割任务
_C.TRAIN.LANE_ONLY = False #是否仅训练车道线分割任务
_C.TRAIN.DET_ONLY = False #是否仅训练检测任务
_C.TRAIN.PLOT = False #是否绘制信息

#测试相关参数
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 32 #单个GPU上的batch size
_C.TEST.MODEL_FILE = '' #模型路径
_C.TEST.SAVE_JSON = False #是否保存json文件
_C.TEST.SAVE_TXT = False #是否保存txt文件
_C.TEST.PLOTS = False #是否绘制信息
_C.TEST.NMS_CONF_THRESHOLD = 0.001 #NMS置信度阈值
_C.TEST.NMS_IOU_THRESHOLD = 0.6 #NMS IoU阈值

#更新配置文件
def update_config(cfg, args):
    cfg.defrost()
    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir
    if args.logDir:
        cfg.LOG_DIR = args.logDir
    cfg.freeze()
