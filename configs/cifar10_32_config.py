config = {}
config["batch_size"] = 256
config["epochs"] = 100             # number of epochs to train (default: 10)
config["no_cuda"] = False         # disables CUDA training
config["seed"] = 1265
config["image_size"] = 32
config["log_interval"] = 1     # how many batches to wait before logging training status
config["learning_rate"] = 0.06
config["momentum"] = 0.1
config["gamma"] = 0.99
config["weight_decay"] = 0.0

config["data_set"] = "CIFAR10"
config["num_classes"] = 10
config["num_features"] = 512