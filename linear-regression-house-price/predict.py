from sklearn import linear_model

def csv_to_array(filename):
	result = []
	with open(filename, 'r') as fp:
		for line in fp:
			if line.startswith('#'): continue
			cols = line.strip().split(',')
			result.append(cols)
	return result

data = csv_to_array('houseprice.csv')
X = []
y = []

for i in data:
	y.append(int(i[0]))
	X.append(map(int, i[1:]))

clf = linear_model.LinearRegression()
clf.fit(X, y)

print 'Coefficient:', clf.coef_

# Try to predict the price
# Colums are: Year of Construction, # of rooms, # of bedrooms, # of bathrooms, # of power rooms, # of garages
print clf.predict([1966, 9, 3, 1, 1, 1])