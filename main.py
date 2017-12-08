from lib.BatchDeconvNet import BatchDeconvNet
from lib.DeconvNet import DeconvNet
from lib.SegNet import SegNet

def main():
  net = SegNet()
  net.train()

if __name__ == "__main__":
  main()