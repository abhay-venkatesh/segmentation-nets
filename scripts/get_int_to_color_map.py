import json

clsToInt = {}
with open('../dat/reducedClassesToInt.json') as infile:
	clsToInt = json.load(infile)

colorToCls = {}
with open('../dat/reducedColorsToClasses.json') as infile:
	colorToCls = json.load(infile)

# have
# class -> int
# color -> class
# color -> int

numToColor = {}
for color in colorToCls:
	num = clsToInt[colorToCls[color]]
	if num != 0:
		numToColor[num] = color

with open('../dat/reducedIntToColors.json', 'w') as outfile:
	colorToCls = json.dump(numToColor, outfile)