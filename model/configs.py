from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.MLP_LAYER = 3
_C.MODEL.DROPOUT = 0.1
_C.MODEL.lr_decay = 0.5
_C.MODEL.decay_interval = 10
_C.MODEL.NUM_EPOCHS = 40
_C.MODEL.BATCH_SIZE = 4
_C.MODEL.LR = 0.0001
_C.MODEL.WEIGHT_DECAY = 1e-4
_C.MODEL.SEED = 2048
_C.MODEL.KFS = 0
_C.MODEL.GAT_HEADS = 4

_C.MODEL.NUM_ATOM_TYPES = 10
_C.MODEL.NUM_FINGERPRINT_TYPES = 1000
_C.MODEL.NUM_PHYSICOCHEMICAL = 6 #5 proxy + molar mass + weight

_C.SUB = CN()
_C.SUB.RADIUS = 2
_C.SUB.NGRAM = 3
_C.SUB.DIM = 64
_C.SUB.LAYER_GNN = 3
_C.SUB.LAYER_NN = 2
_C.SUB.LAYER_OUTPUT = 1

#融合
_C.FUSION = CN()
_C.FUSION.METHOD = "early_concat"   # "early_concat"  "gated_fusion"  "attn_key"

_C.DIR = CN()
_C.DIR.DATASET = 'your dataset csv file'
_C.DIR.PREPROCESSED_DIR = 'preprocessed pkl file'
_C.DIR.OUTPUT_DIR = "your output dir path"

def get_cfg_defaults():
    return _C.clone()
