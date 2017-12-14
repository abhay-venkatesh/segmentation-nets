from lib.SegNet import SegNet
from lib.BatchSegDeconvNet import BatchSegDeconvNet

def main():
  dataset_directory = './datasets/unreal_randomyaw/'
  net = BatchSegDeconvNet(dataset_directory, batch_size=5)
  net.train(num_iterations=50000)

if __name__ == "__main__":
  main()
