from lib.BatchSegNet import BatchSegNet
from lib.DFSegNet import DFSegNet

def main():
  dataset_directory = './datasets/unreal_randomyaw/'
  net = DFSegNet(dataset_directory)
  # net = BatchSegNet(dataset_directory)
  net.train(num_iterations=100000, learning_rate=1e-2)

if __name__ == "__main__":
  main()
