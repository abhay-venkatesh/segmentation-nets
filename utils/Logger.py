import os
import csv
if os.name != 'nt':
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
else:
  import matplotlib.pyplot as plt
	

class Logger:

  def __init__(self, session=1):
    if not os.path.exists('./logs/'):
      os.makedirs('./logs/')
    self.session = session

  def log(self, message):
    if not os.path.exists('./logs/logfile-' + str(self.session)):
      with open('./logs/logfile-' + str(self.session), 'w') as outfile:
        outfile.write(message)
    else:
      with open('./logs/logfile-' + str(self.session), 'a') as outfile:
        outfile.write(message)

  def log_for_graphing(self, iterations, loss, accuracy, mean_IoU):
    if not os.path.exists('./logs/logfile-graphing-' + str(self.session)):
      with open('./logs/logfile-graphing-' + str(self.session), 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([iterations, loss, accuracy, mean_IoU])
    else:
      with open('./logs/logfile-graphing-' + str(self.session), 'a', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([iterations, loss, accuracy, mean_IoU])

  def log_for_test_graphing(self, sequence_id, accuracy):
    """
    For 20-Sequence test.
    """
    if not os.path.exists('./logs/logfile-test-graphing-' + str(self.session)):
      with open('./logs/logfile-test-graphing-' + str(self.session), 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([sequence_id, accuracy])
    else:
      with open('./logs/logfile-test-graphing-' + str(self.session), 'a', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([sequence_id, accuracy])

  def graph_training_stats(self):
    if not os.path.exists('./logs/logfile-graphing-' + str(self.session)):
      pass
    else:
      with open('./logs/logfile-graphing-' + str(self.session)) as infile:
        reader = csv.reader(infile, delimiter=",")
        iterations = []
        losses = []
        accuracies = []
        mean_IoUs = []
        for row in reader:
          iterations.append(int(row[0]))
          losses.append(float(row[1]))
          accuracies.append(float(row[2]))
          mean_IoUs.append(float(row[3]))

        if not os.path.exists('./metrics/'):
          os.makedirs('./metrics/')

        plt.figure(0)
        plt.plot(iterations, losses)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.savefig('./metrics/iterations_vs_loss.png')

        plt.figure(1)
        plt.plot(iterations, accuracies)
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.savefig('./metrics/iterations_vs_accuracy.png')

        plt.figure(2)
        plt.plot(iterations, mean_IoUs)
        plt.ylabel('mean_IoU')
        plt.xlabel('Iterations')
        plt.savefig('./metrics/iterations_vs_mean_IoU.png')

  def graph_test_stats(self):
    if not os.path.exists('./logs/logfile-test-graphing-' + str(self.session)):
      pass
    else:
      with open('./logs/logfile-test-graphing-' + str(self.session)) as infile:
        reader = csv.reader(infile, delimiter=",")
        sequence_ids = []
        accuracies = []
        for row in reader:
          sequence_ids.append(int(row[0]))
          accuracies.append(float(row[1]))

        if not os.path.exists('./metrics/'):
          os.makedirs('./metrics/')
        plt.figure(3)
        plt.plot(sequence_ids, accuracies)
        plt.ylabel('Accuracy')
        plt.xlabel('Perspective')
        plt.savefig('./metrics/iterations_vs_sequence_ids.png')

    

def main():
  logger = Logger()
  logger.graph_training_stats()
  logger.graph_test_stats()

if __name__ == "__main__":
  main()





  
