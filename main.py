from lib.BatchSegNet import BatchSegNet
from lib.DFSegNet import DFSegNet

def main():
  dataset_directory = './datasets/UnrealNeighborhood-11Class-StreetPrimary-0.15/'
  net = DFSegNet(dataset_directory)
  # net = BatchSegNet(dataset_directory)
  net.train(num_iterations=100000, learning_rate=1e-2)

if __name__ == "__main__":
  main()
