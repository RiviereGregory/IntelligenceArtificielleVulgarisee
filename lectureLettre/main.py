from mnist import MNIST

# Chargement des images
emnist_data = MNIST(path='datas\\', return_type='numpy')
emnist_data.select_emnist('letters')
Images, Libelles = emnist_data.load_training()
