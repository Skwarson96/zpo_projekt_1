# zpo_projekt_1

The project concerned the recognition of places in Poznań with the Bag of Visual Words.

#Example photos used in the project 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Wie%C5%BCowiec_Ba%C5%82tyk_w_Poznaniu.jpg/1024px-Wie%C5%BCowiec_Ba%C5%82tyk_w_Poznaniu.jpg" width="350">
(source: wikipedia)

## Results:
- private test set: 96%
- final, unknown test set: 82.5%

## Usage:
- `data_augmentation.py` - generate augmentated data (rotation, perpendicularly flip, lightening, darkening, add gaussian_noise and change saturation)
- `projekt.py` - train model and classifier 
- `main.py` - test model and classifier 






