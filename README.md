# zpo_projekt_1

The project concerned the recognition of places in Poznań with the Bag of Visual Words.

#Example photos used in the project 

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5a/POL_Pozna%C5%84_Okr%C4%85glak.jpg" width="350">

(source: wikipedia)

## Results:
- private test set: 96%
- final, unknown test set: 82.5%

## Usage:
- `data_augmentation.py` - generate augmentated data (rotation, perpendicularly flip, lightening, darkening, add gaussian_noise and change saturation)
- `projekt.py` - train model and classifier 
- `main.py` - test model and classifier 






