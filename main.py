from lib.SegNet import SegNet
from lib.BatchSegDeconvNet import BatchSegDeconvNet
from lib.BatchSegNet import BatchSegNet

def main():
  dataset_directory = './datasets/unreal_randomyaw/'
  net = BatchSegNet(dataset_directory)
  # net.train(num_iterations=80000, learning_rate=1e-2)
  # net.test()
  net.test_outside_dataset()

if __name__ == "__main__":
  main()
