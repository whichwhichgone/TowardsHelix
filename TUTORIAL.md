# 统一数据集格式到OXE格式

## 在使用自己的数据训练模型时，需要进行以下步骤

> **Note**: 整个模型构造训练数据的入口函数为`train.py`文中的`get_vla_dataset_and_collator`。在该函数中首先定义了`RLDSBatchTransform`和`PaddedCollatorForActionPrediction`两个数据类分别用于对最后喂入模型的数据进行统一格式化的处理和对齐数据长度。然后，该函数中同时定义了一个`RLDSDataset`类的实例来管理训练模型用到的所有数据。RLDSDataset继承自`tf.data.Dataset`，因此可以直接使用`tf.data.Dataset`的API来对数据进行处理。

**Step 1**: 首先需要修改`configs.py`文件中的`OXE_DATASET_CONFIGS`字典（假定自己的数据已经转为了rlds的标准形式），该字典中包含了所有的OXE数据集，以及每个数据集在训练时使用的各个字段的数据格式。`make_oxe_dataset_kwargs`函数将会使用该配置中定义的数据集字段来构造数据集相关的属性参数，并用于模型后续的训练。

**Step 2**: 同时还要注意修改一些对数据集进行预处理转换的函数，参考`OXE_STANDARDIZATION_TRANSFORMS`定义的各个函数，比如`berkeley_autolab_ur5_dataset_transform`函数。

**Step 3**: 利用`make_dataset_from_rlds`函数来将事先准备好的rlds格式的数据集转换为**Step 1**中所期望的数据集格式。该函数在`make_interleaved_dataset`函数执行过程中被调用，转换之后的数据集中数据的格式为：

> **Note**: `make_interleaved_dataset`是另外一个重要的数据构造函数，负责具体的项目中`RLDSDataset`类实例的构建。

```python
{
"observation": {
    {
        "image_primary": xxx;
        "image_secondary": xxx;
        ...;
        "depth_primary": xxx;    # 目前没有使用这个字段，可根据需要添加
        ...;
        "proprio": xxx;          # 目前没有使用这个字段，可根据需要添加
        "timestep";
    }
};
"task": {"language_instruction": xxx};
"action": xxx;
"dataset_name": xxx;
"absolute_action_mask": xxx;
}
```

**Step 4**: 数据处理流程中，使用了`apply_trajectory_transforms`函数来在traj层级对训练数据进行处理，使用了`apply_frame_transforms`函数来在frame层级对训练数据进行处理。

**Step 5**: `PaddedCollatorForActionPrediction`数据类用于最后组装参与训练的batch数据，对齐batch里不同数据的长度。 `RLDSBatchTransform()`则在每个batch数据喂入模型前进行最后格式化处理。
