from lib.BatchDeconvNet import BatchDeconvNet
from lib.DeconvNet import DeconvNet
from lib.SegNet import SegNet
from lib.BatchSegNet import BatchSegNet

def main():
  net = BatchSegNet()
  net.train()

if __name__ == "__main__":
  main()