# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
from .custom_lgc import CustomDataset_lgc


@DATASETS.register_module()
class LGCDataset(CustomDataset_lgc):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    # CLASSES = ('tree', 'farmland','grass', 'bush','clutter')
    #
    # PALETTE = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255,255,0],[255, 255, 255]]

    CLASSES = ('clutter', 'farmland','grass', 'tree','bush')

    PALETTE = [[255, 255, 255], [0, 0, 255], [255, 0, 0], [255,255,0],[0, 255, 0]]

    def __init__(self, **kwargs):
        super(LGCDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            #ignore_index=4,
            **kwargs)
