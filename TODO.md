# Project TODO List

## Overall Goals

- [ ] Run traditional classifiers (KNN, LR, RF, SVM) against clean/poisoned data
- [ ] Compare performance between clean and poisoned datasets
- [ ] Analyze feature space differences between clean and poisoned samples
- [ ] Visualize results and create comprehensive analysis

## Immediate Tasks

1. [ ] Test Imagenette pipeline with small sample size (500 samples)
2. [ ] Implement proper feature visualization for analysis
3. [ ] Add support for poisoned data loading and comparison

## GTSRB Tasks (Paused)

- [ ] Train GTSRB model to get better feature extraction (currently 3-4% accuracy with untrained model)
- [ ] Create or obtain poisoned GTSRB dataset
- [ ] Run full GTSRB experiments with trained model
- [ ] Compare clean vs poisoned performance

## Infrastructure Improvements

- [ ] Add checkpoint saving/loading for trained models
- [ ] Implement proper progress tracking during feature extraction
- [ ] Add experiment history tracking
- [ ] Improve logging configuration (currently seeing duplicate initialization)
- [ ] Add proper error handling for dataset loading

## Analysis & Visualization

- [ ] Implement t-SNE/UMAP visualization of feature spaces
- [ ] Add confusion matrix visualization
- [ ] Create performance comparison plots
- [ ] Generate statistical analysis of classifier performance

## Documentation

- [ ] Document dataset preparation process
- [ ] Add instructions for running experiments
- [ ] Create analysis pipeline documentation
- [ ] Add requirements and setup instructions

## Future Enhancements

- [ ] Add support for more datasets
- [ ] Implement additional classifiers if needed
- [ ] Add cross-validation support
- [ ] Create automated test suite
- [ ] Add experiment configuration validation
