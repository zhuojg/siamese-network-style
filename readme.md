# Model
style_model.py - model code
You don't need to run this file. But if you run this file, you can get the features of '404.jpg'.

# Offline
job2.py - calculate images' style features using model in style_model.py and reduce dimension using PCA (and t-SNE in the comments). 
Remember to change the image_path and result_path!
Run this file to get the features for clustering.

# Online
demo.py - cluster(kmeans) images using the features calculated by the job2.py(stored in cnn_features.pkl or pca_features.pkl) and save the results in pictures.
Remember to change the feature_path and save_path!
