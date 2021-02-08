# zpo_projekt_1

The project concerned the recognition of places in Poznań with the Bag of Visual Words.

#Example photos used in the project 
<img src="https://pl.wikipedia.org/wiki/Uniwersytet_im._Adama_Mickiewicza_w_Poznaniu" width="350">
<img src="https://pl.wikipedia.org/wiki/Bazylika_archikatedralna_%C5%9Awi%C4%99tych_Aposto%C5%82%C3%B3w_Piotra_i_Paw%C5%82a_w_Poznaniu" width="350">
<img src="https://pl.wikipedia.org/wiki/Wie%C5%BCowiec_Ba%C5%82tyk" width="350">
<img src="https://pl.wikipedia.org/wiki/Teatr_Wielki_im._Stanis%C5%82awa_Moniuszki_w_Poznaniu#/media/Plik:Teatr_Wielki,_Poznan,_Polonia,_2014-09-18,_DD_53.jpg" width="350">
<img src="https://pl.wikipedia.org/wiki/Okr%C4%85glak_w_Poznaniu" width="350">
(source: wikipedia)

## Results:
- private test set: 96%
- final, unknown test set: 82.5%

## Usage:
- `data_augmentation.py` - generate augmentated data (rotation, perpendicularly flip, lightening, darkening, add gaussian_noise and change saturation)
- `projekt.py` - train model and classifier 
- `main.py` - test model and classifier 






