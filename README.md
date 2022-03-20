# Conditional-WGAN-GP
From scratch implementation of Conditional WGAN with Gradient Penalty

# Pre-requisites
Install all required dependencies into your local environment by running the following command
> pip install -r requirements.txt

# Important Note
- These .py files were created to run the training on your local machines, for any changes related to the type of dataset selected(In case one wishes to use their own
dataset) you may change the DATASET variable in controller.py and add the location for the dataset in DATASET_LOCATION in the same file.

- By default, this program will train on MNIST dataset.

- The functionality for save and loading the generator and critic is not available since this is an intuitive look into the architecture and 
is created from scratch for the same reason.

- For a view of the results after training on MNIST dataset, kindly view the Results/Conclusion section in Colab Notebook.

# Colab Notebook

- The dependencies are pre-loaded

- The notebook is an educational guide into how each element(cell) works and its purpose, the markdown cells also have hyperlinks for one's further exploration into
  this architecture.
  
# Acknowledgments

Special thanks to Aladdin Persson who has a YouTube channel on which he briefly explains and develops Machine Learning models.
