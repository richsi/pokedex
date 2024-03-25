# Pokedex
Generation 1 starter Pokemon classifier - Charmander, Squirtle, Bulbasaur

## Workflow
* scraper.py: Uses Bing Image Search API to scrape internet for pokemon images. Stores images into respective dataset folder
* model/train.py: Uses PyTorch to transform and load the images. It also trains the custom VGGNet model (model/model.py) using CrossEntropy as the criterion and Adam for the optimizer. Model is then saved in model/model.pt and the model's loss and accuracy are saved into plot.png
* classify.py:

