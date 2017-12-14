from lib.SegNet import SegNet
from lib.BatchSegDeconvNet import BatchSegDeconvNet
from lib.BatchSegNet import BatchSegNet


def main():
  dataset_directory = './datasets/unreal_randomyaw/'
  net = BatchSegNet(dataset_directory)
  net.train(num_iterations=50000)

if __name__ == "__main__":
  main()
