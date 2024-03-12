# Learning place representations from spatial interactions
## Abstract
The development of geospatial artificial intelligence (GeoAI) systems depends on the ability to learn effective representations of places. To learn accurate place representations from spatial interactions, it is important to extract features that capture both the spatial and non-spatial driving factors. However, existing methods lack a robust interpretation and the explanatory power of the learned representations on spatial factors remains unexplored. Here, we propose an approach to learning place representations from spatial interactions. Our method is inspired by flow allocation, which is the main focus of single-constrained gravity models. We first validate the method on synthetic flows with known driving factors and then apply it to multi-scale real-world flows. Results show that the learned representations can effectively capture features that explain place characteristics, along with the impact of spatial impedance. Our study not only contributes an efficient method to learn place representations from spatial interactions but also offers insights into pre-training procedures in GeoAI.

![image](https://github.com/Xuechen0123/FlowAlloc_SI_placeemb/blob/main/img/Fig1_Schematic_of_the_workflow.png)

# Basic Usage

## Preparation

Install the required packages with

```
conda env create -f environment.yml
```

## Training

To run our method with example data, execute the following command from the project home directory:

```
python3 src/main.py --num-places=1000 --representation-save-file=emb/embeddings.pkl --place-embed-dim=32 --gpu=0 --lr=1e-3 --adam-beta1=0.9 --adam-beta2=0.999 --wd=1e-2 --batch-size=32 --num-epochs=500 --warmup-epochs=10 --print-freq=100 --optimize-out --optimize-in
```

## Output file

Place representations are saved in the specified save folder in .pkl format. 

Run pickle.load to access the saved result which is organized as python dict:

- 'place_id': list of place ids, origin/destination are organized according to the place_id_list

- 'origin_representation': array of len(place_id_list) $\times$ embedding dim

- 'destination_representation': array of len(place_id_list) $\times$ embedding dim
