# U-pre
Using U-net1D to achieve significant results in time series prediction

This project, named U-pre, is led by Selen at Xidian and focuses on time series prediction. After investigating time series forecasting tasks, Selen identified that the input and output of time series prediction exhibit the same distribution, high correlation, and identical dimensionality, which align perfectly with the requirements for using U-Net with Conv1D. Leveraging this insight, Selen applied U-Net1D to the ETT series datasets for forecasting. While the results did not surpass current state-of-the-art (SOTA) methods, they outperformed several 2022 baselines. Consequently, the project was made publicly available. Selen is eager to encourage collaborative efforts to enhance this project with supplementary experiments and auxiliary measures, exploring the potential for U-pre to evolve into a new SOTA solution. 

# U-pre v2

A significant limitation of U-pre lies in Conv1D's inability to capture long-term dependencies, as it focuses solely on short-term relationships within each window. However, this is not without value, as it complements transformers by addressing local temporal details that might be overlooked in their focus on long-range dependencies. To overcome this limitation, U-pre V2 introduces an innovative hybrid approach that integrates the output of a U-Net convolutional encoder with the original time series as input tokens to a single BERT encoder layer. This approach leverages both local and global contextual information. Furthermore, the U-Net's preprocessing stage implicitly assigns local gradient weights to different timesteps, enabling a more nuanced attention mechanism that transcends the self-attention limitations of traditional sequence models, thereby capturing richer spatio-temporal dependencies.

# Experiment

*ETT数据集上的预测性能（MSE/MAE）比较 (96步),我们发布了[中文报告](https://github.com/Selen-Suyue/Selen-Suyue.github.io/raw/master/files/upre.pdf)*

| 模型 | ETTm1 (MSE/MAE) | ETTm2 (MSE/MAE) | ETTh1 (MSE/MAE) | ETTh2 (MSE/MAE) | 平均 (MSE/MAE) |
|------|-----------------|-----------------|-----------------|-----------------|----------------|
| iTransformer | 0.334/0.368 | 0.180/0.264 | 0.386/0.405 | 0.297/0.349 | 0.299/0.347 |
| RLinear | 0.355/0.376 | 0.182/0.265 | 0.386/0.395 | 0.288/0.338 | 0.303/0.344 |
| PatchTST | 0.329/0.367 | 0.175/0.259 | 0.414/0.419 | 0.302/0.348 | 0.305/0.348 |
| Crossformer | <span style="color:red">0.404/0.426</span> | <span style="color:red">0.287/0.366</span> | <span style="color:red">0.423/0.448</span> | <span style="color:red">0.340/0.374</span> | <span style="color:red">0.364/0.404</span> |
| TiDE | 0.364/0.387 | <span style="color:red">0.207/0.305</span> | 0.384/0.402 | <span style="color:red">0.340/0.374</span> | 0.324/0.367 |
| TimesNet | 0.338/0.375 | 0.187/0.267 | <span style="color:red">0.479/0.464</span> | <span style="color:red">0.400/0.440</span> | <span style="color:red">0.351/0.387</span> |
| DLinear | 0.345/0.372 | <span style="color:red">0.193/0.292</span> | 0.386/0.400 | <span style="color:red">0.402/0.414</span> | <span style="color:red">0.332/0.369</span> |
| SCINet | <span style="color:red">0.418/0.438</span> | <span style="color:red">0.286/0.377</span> | <span style="color:red">0.654/0.599</span> | <span style="color:red">0.376/0.419</span> | <span style="color:red">0.434/0.458</span> |
| FEDformer | <span style="color:red">0.379/0.419</span> | <span style="color:red">0.203/0.287</span> | <span style="color:red">0.513/0.491</span> | <span style="color:red">0.449/0.459</span> | <span style="color:red">0.386/0.414</span> |
| Stationary | <span style="color:red">0.386/0.398</span> | <span style="color:red">0.192/0.274</span> | <span style="color:red">0.449/0.459</span> | <span style="color:red">0.526/0.516</span> | <span style="color:red">0.388/0.412</span> |
| Autoformer | <span style="color:red">0.505/0.475</span> | <span style="color:red">0.255/0.339</span> | <span style="color:red">0.449/0.459</span> | <span style="color:red">0.450/0.459</span> | <span style="color:red">0.415/0.433</span> |
| U-preV1 | <span style="color:red">0.466/0.451</span> | <span style="color:red">0.195/0.275</span> | <span style="color:red">0.524/0.483</span> | <span style="color:red">0.367/0.395</span> | <span style="color:red">0.388/0.401</span> |
| U-preV2 | 0.370/0.396 | 0.188/0.273 | 0.419/0.429 | 0.325/0.371 | 0.326/0.367 |
