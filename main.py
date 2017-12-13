from lib.BatchDeconvNet import BatchDeconvNet
from lib.DeconvNet import DeconvNet
from lib.SegNet import SegNet
from lib.BatchSegNet import BatchSegNet

def main():
  dataset_directory = './datasets/unreal_randomyaw/'
  net = BatchSegNet(dataset_directory)
  net.train(num_iterations=50000, batch_size=5)

if __name__ == "__main__":
  main()
