
## Run training and predicitons on the primary cubelet:

IMPORTANT: set `primary_only=True` in generate_dataset.py

`python generate_dataset.py --data-dir ../Data/PrimaryOnly/Uniform --out PrimaryOnlyUniform_Epos.pt --target Epos`  # or "energy,centroid"  or  Edsp ...

`python train_model.py --cache PrimaryOnlyUniform_Epos.pt --epochs 5 --lr 1e-2 --model-out snn_PrimaryTrained_Epos.pth`

`python print_predictions.py --cache PrimaryOnlyUniform_Epos.pt --model snn_PrimaryTrained_Epos.pth`

## Run training in primary cubelet and predict in all:

IMPORTANT: set `primary_only=False` in generate_dataset.py

`python generate_dataset.py --data-dir ../Data/All_small --out All_small_Epos.pt --target Epos`

Assuming we already trained a model called `snn_PrimaryTrained_Epos`with primary only data, we can test it on all cubelets data:

`python print_predictions.py --cache All_small_Epos.pt --model snn_PrimaryTrained_Epos.pth`

