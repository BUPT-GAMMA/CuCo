## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)

Then, you need to create a directory for recoreding  results to avoid errors:

```
mkdir logs
```

## Training & Evaluation

```
python cuco.py --DS $DATASET_NAME --lr 0.01 --local --num-gc-layers 3 --batch_size 128  --aug $AUGMENTATION 
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/),  ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.






