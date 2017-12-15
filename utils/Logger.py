import os

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
      with open('./logs/logfile-graping-' + str(self.session), 'w') as outfile:
        message = str(iterations) + ',' + str(loss) + ',' + str(accuracy) + '\n'
        outfile.write(message)
    else:
      with open('./logs/logfile-graping-' + str(self.session), 'a') as outfile:
        outfile.write(message)



  