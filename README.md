# Fei-Fei-Huang

## fashion-mnist prediction

### 经过调参，准确率已经达到 92.8%；

### 网络参数：
    ```
    
    params = {
            'datasets': data, 
            'growth_rate': 12, 
            'depth': 40, 
            'total_blocks': 3, 
            'keep_prob': 0.8, 
            'weight_decay': 0.001, 
            'nesterov_momentum': 1 , 
            'model_type': 'DenseNet_BC', 
            'dataset_name': 'fashion-mnist', 
            'should_save_logs': True, 
            'should_save_model': True,
    }
    
    ```

### 训练参数：
    ```
    
    train_params = {
            'n_epochs': 300,
            'initial_learning_rate': 0.001,
            'batch_size': 8,
            'reduce_lr_epoch_1': 100,
            'reduce_lr_epoch_2': 200,
            'validation_set': True
    }
    
    ```
    
![Image](https://i.imgur.com/h8aFfgZ.png)
