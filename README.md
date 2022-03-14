#  Adaptive Spatio-Temporal Graph Learning for Bus Station Profiling
This is the implementation of SPA based on PyTorch. 
The SPA adopts and advances the graph learning structure through a few novel ideas: (1) Designing an adaptive graph learning mechanism to capture the complex and dynamic spatio-temporal dependencies rather than relying on predefined spatio-temporal graphs; (2) Modeling spatio-temporal interactions in shifted spatial graphs to learn fine-grained spatio-temporal features; (3) Employing self-attention mechanism to learn the long-term temporal dependencies preserved in mobility data. We conduct extensive experiments on three real-world spatio-temporal datasets. 
## structure of the code:  

- `lib` folder: some methods for data loading and processing from [AGCRN](https://github.com/LeiBAI/AGCRN); 
- `utils.py`: method of loading adjacency graph;  
- `model.py`: implementation of SPA;  
- `train.py`, `run.py`: train and run the model.   
 
You can use `python run.py --dataset PeMSD4 --num_nodes 370` command to run the code.

