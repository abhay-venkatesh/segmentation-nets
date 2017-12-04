"""
Create train.txt and val.txt files
"""

def createTrainAndVal():
	with open('train.txt', 'w') as trainfile:
		with open('val.txt', 'w') as valfile:

			numTrainingImages = 2800
			numValidationImages = 1200

			for i in range(numTrainingImages):
				trainfile.write('pic' + str(i) + ".png\n")

			for i in range(numValidationImages):
				valfile.write('pic' + str(i + numTrainingImages) + ".png\n")

def createTest():
	with open('test.txt', 'w') as testfile:

		numTestImages = 20
		for i in range(numTestImages):
			testfile.write('pic' + str(i) + ".png\n")

def main():
	createTest()

if __name__ == "__main__":
	main()
