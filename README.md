# Anti Drowning Detection Algorithm

Recently, the frequency of drowning incidences has been increasing. A study revealed children aged 1-4 and adolescents to be most at risk of death in swimming pools and freshwater respectively in the UK. Current technologies like Poseidon detect, using cameras, when a person has been motionless at the bottom of a pool for 10 seconds, and sends an alarm. Other technologies rely on manual deployment of flotation devices which is unsuitable for children who will panic in a crisis. We aim to create a device that automatically detects if a person is drowning and auto inflates to help the user to safety. This repository contains a suite of trained convolutional neural networks (CNNs) to differentiate between swimming strokes and drowning (flailing) along with our thresholding algorithm to detect sharp instantaneous changes in heart rate, which may be used in such a device. We obtain accuracies of 97% for both CNNs.


### Prerequisites

Install the following dependencies to your virtual environment:

```
pip install scikit-learn
pip install pywt
pip install tensorflow
pip install keras
```

### Results
![cnn results](cnn_results.png) ![confusion results](confusion_matrix.png)

![video_mdm3_3](https://user-images.githubusercontent.com/50496437/118116182-dce5e700-b3e1-11eb-9d3a-881d869aa761.gif)


## Authors

* **Jake Beard** - [GitHub Page](https://github.com/jake-beardo)
* **Vedang Joshi** - [GitHub Page](https://github.com/vedang-joshi)
* **Isobel Russell** - [GitHub Page](https://github.com/isobelrussell00)
* **Will Womersley** - [GitHub Page](https://github.com/WWomersley)
* **Will Tipping** - [GitHub Page](https://github.com/WillTipping)

## License

This project is licensed under the GNU General Public License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* A special thanks to [Mark Blyth](https://research-information.bris.ac.uk/en/persons/mark-d-blyth) for guiding us through this project.
