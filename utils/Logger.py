import os
import csv
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

  def log_for_graphing(self, iterations, loss, accuracy):
    if not os.path.exists('./logs/logfile-graphing-' + str(self.session)):
      with open('./logs/logfile-graphing-' + str(self.session), 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([iterations, loss, accuracy])
    else:
      with open('./logs/logfile-graphing-' + str(self.session), 'a', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([iterations, loss, accuracy])

  def graph_training_stats(self):
    if not os.path.exists('./logs/logfile-graphing-' + str(self.session)):
      pass
    else:
      with open('./logs/logfile-graphing-' + str(self.session)) as infile:
        reader = csv.reader(infile, delimiter=",")
        iterations = []
        losses = []
        accuracies = []
        for row in reader:
          iterations.append(int(row[0]))
          losses.append(float(row[1]))
          accuracies.append(float(row[2]))

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

def main():
  logger = Logger()
  logger.graph_training_stats()

if __name__ == "__main__":
  main()





  